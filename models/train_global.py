import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from config import TARGET_FS, SEQ_LEN, SPLIT_SEED, VAL_RATIO, SPLIT_INFO_PATH
from dataset import DataLoader as ECGDataLoader
from dataset import Preprocessor
from .model import ECGReconstructionModel

import warnings
warnings.filterwarnings('ignore')

# Konfiguracja urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _augment_batch(pcg: torch.Tensor, scg: torch.Tensor) -> tuple:
    """
    Prosta augmentacja danych in-batch:
      - Skalowanie amplitudy: losowy mnożnik ∈ [0.9, 1.1] per próbka
      - Szum gaussowski: σ = 0.02 (proporcjonalnie małe względem z-score)
    Stosowana tylko do wejść (PCG/SCG), nie do celu EKG.
    """
    B = pcg.size(0)
    amp = torch.empty(B, 1, 1, device=pcg.device).uniform_(0.9, 1.1)
    noise_p = torch.randn_like(pcg) * 0.02
    noise_s = torch.randn_like(scg) * 0.02
    return pcg * amp + noise_p, scg * amp + noise_s


def train_epoch(model, train_loader, optimizer, criterion, augment: bool = False):
    model.train()
    total_loss = 0
    valid_batches = 0
    for batch_idx, (pcg, scg, target_ecg) in enumerate(train_loader):
        pcg, scg, target_ecg = pcg.to(device), scg.to(device), target_ecg.to(device)

        # Pomijaj batch jesli wejscie zawiera NaN (np. z wavelet na zdegenerowanych oknach)
        if torch.isnan(pcg).any() or torch.isnan(scg).any() or torch.isnan(target_ecg).any():
            continue

        if augment:
            pcg, scg = _augment_batch(pcg, scg)

        optimizer.zero_grad()
        output_ecg = model(pcg, scg)
        loss = criterion(output_ecg, target_ecg)

        # Pomijaj batch jesli strata jest NaN - zapobiega zakazeniu wag modelu
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

    return total_loss / valid_batches if valid_batches > 0 else float('nan')

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    all_correlations = []
    with torch.no_grad():
        for pcg, scg, target_ecg in val_loader:
            pcg, scg, target_ecg = pcg.to(device), scg.to(device), target_ecg.to(device)
            output_ecg = model(pcg, scg)
            
            loss = criterion(output_ecg, target_ecg)
            total_loss += loss.item()
            
            # Wektoryzacja danych do policzenia korelacji Pearsona
            out_np = output_ecg.cpu().squeeze(-1).numpy()
            tar_np = target_ecg.cpu().squeeze(-1).numpy()
            for i in range(out_np.shape[0]):
                corr = np.corrcoef(out_np[i], tar_np[i])[0, 1]
                if not np.isnan(corr):
                    all_correlations.append(corr)
                    
    avg_corr = np.mean(all_correlations) if all_correlations else 0
    return total_loss / len(val_loader), avg_corr

def main(model_name='bilstm_transformer', pipeline_name='kaisti', epochs=50, batch_size=32, resume=True, force_preprocess=False):
    loader = ECGDataLoader()
    pre = Preprocessor(fs=TARGET_FS)

    # ── Auto-konfiguracja dla nowych modeli ──────────────────────────────────
    # Wartości domyślne (stare zachowanie – bez zmian dla istniejących modeli)
    weight_decay = 1e-5
    patience     = None      # None = brak early stopping
    augment      = False
    stride       = None      # None = brak nakładania okien
    balance      = True      # True = przycinanie do min_n (stare zachowanie)
    seq_len      = SEQ_LEN

    # Słownik potoków do dynamicznego wyboru
    from pipelines import kaisti_pipeline, advanced_filtering_pipeline, aggregate_and_balance_datasets, aggregate_balanced_sources
    from pipelines import ALTERNATIVE_PIPELINES

    PIPELINES = {
        'kaisti':    kaisti_pipeline,
        'advanced':  advanced_filtering_pipeline,
        **ALTERNATIVE_PIPELINES
    }

    pipeline_fn = PIPELINES.get(pipeline_name, kaisti_pipeline)
    print(f"Używam potoku preprocessingu: '{pipeline_name}'")
    
    # Obsługa pamięci podręcznej (cache) preprocessingu
    os.makedirs("data/cache", exist_ok=True)
    train_cache_path = f"data/cache/train_{pipeline_name}.npz"
    val_cache_path = f"data/cache/val_{pipeline_name}.npz"
    
    loaded_from_cache = False
    train_data = None
    val_data = None
    
    if not force_preprocess and os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        print(f"[CACHE] Odnaleziono zapreprocesowane dane dla '{pipeline_name}'. Wczytywanie...")
        try:
            train_npz = np.load(train_cache_path, allow_pickle=True)
            val_npz = np.load(val_cache_path, allow_pickle=True)
            train_data = {k: train_npz[k] if k in train_npz else None for k in ['gcg_final', 'scg_final', 'ecg_final']}
            val_data = {k: val_npz[k] if k in val_npz else None for k in ['gcg_final', 'scg_final', 'ecg_final']}
            loaded_from_cache = True
            print("[CACHE] Pomyślnie wczytano dane z pamięci podręcznej!")
        except Exception as e:
            print(f"[CACHE] Nie udało się wczytać cache: {e}. Uruchamiam pełny preprocessing...")
            
    if not loaded_from_cache:
        print("========== Wczytywanie wszystkich dostępnych zbiorów danych ==========")
        all_dfs_dict = loader.load_all_datasets()

        # Stratified split 80/20 OSOBNO dla każdego źródła — zapobiega:
        #   (a) data leakage między pacjentami,
        #   (b) losowej dominacji jednego zbioru w zbiorze treningowym.

        # Listy nazw rekordów — w tej samej kolejności co wczytane DataFrame'y
        all_names_dict = {
            'ieee':      loader.list_ieee(),
            'zenodo':    loader.list_zenodo(),
            'physionet': loader.list_physionet() if hasattr(loader, 'list_physionet') else [],
        }

        random.seed(SPLIT_SEED)
        train_dfs_per_source = {}
        val_dfs_per_source   = {}
        split_info = {}
        total_records = 0

        for ds_name, dfs in all_dfs_dict.items():
            if not dfs:
                continue
            names = all_names_dict.get(ds_name, [])
            # Powiązanie nazw z DataFrame'ami przy założeniu identycznej kolejności (load_all_datasets wczytuje dane w kolejności list_*())
            pairs = list(zip(names[:len(dfs)], dfs))
            random.shuffle(pairs)
            n_val = max(1, int(VAL_RATIO * len(pairs)))

            val_pairs   = pairs[:n_val]
            train_pairs = pairs[n_val:]

            val_dfs_per_source[ds_name]   = [df for _, df in val_pairs]
            train_dfs_per_source[ds_name] = [df for _, df in train_pairs]
            split_info[ds_name] = {
                'train': [name for name, _ in train_pairs],
                'val':   [name for name, _ in val_pairs],
            }

            total_records += len(pairs)
            print(f"  [{ds_name}] rekordy: {len(pairs)} "
                  f"→ trening={len(train_pairs)}, walidacja={len(val_pairs)}")

        # Zapisz podział do pliku — można go otworzyć i sprawdzić który rekord gdzie trafił
        try:
            os.makedirs(os.path.dirname(SPLIT_INFO_PATH), exist_ok=True)
            with open(SPLIT_INFO_PATH, 'w', encoding='utf-8') as _f:
                json.dump(split_info, _f, indent=2, ensure_ascii=False)
            print(f"[SPLIT] Podział train/val zapisany → {SPLIT_INFO_PATH}")
        except Exception as e:
            print(f"[SPLIT] Ostrzeżenie: nie udało się zapisać split_info.json: {e}")

        if total_records == 0:
            print("[BŁĄD] Nie znaleziono żadnych danych do treningu.")
            return

        print(f"\n========== Przetwarzanie i balansowanie danych ({total_records} rekordów) ==========")

        # Trening: każde źródło przetwarza się osobno, potem przycięcie do równej liczby okien
        train_data = aggregate_balanced_sources(
            train_dfs_per_source, fs=TARGET_FS, pipeline_func=pipeline_fn,
            seq_len=seq_len, stride=stride, shuffle=True
        )
        # Walidacja: ta sama zasada (równa reprezentacja), bez nakładania okien, bez tasowania
        val_data = aggregate_balanced_sources(
            val_dfs_per_source, fs=TARGET_FS, pipeline_func=pipeline_fn,
            seq_len=seq_len, stride=None, shuffle=False
        )
        
        # Zapis do cache
        if train_data is not None and train_data['scg_final'] is not None:
            try:
                np.savez_compressed(train_cache_path, **{k: v for k, v in train_data.items() if v is not None})
                np.savez_compressed(val_cache_path, **{k: v for k, v in val_data.items() if v is not None})
                print(f"[CACHE] Pomyślnie zapisano przetworzone dane do cache: {train_cache_path}")
            except Exception as e:
                print(f"[CACHE] Ostrzeżenie: Nie udało się zapisać cache: {e}")

    if train_data is None or train_data['scg_final'] is None:
        print("\n[BŁĄD] Nie zebrano żadnych prawidłowych okien do treningu.")
        return
    if val_data is None or val_data['scg_final'] is None:
        print("\n[OSTRZEŻENIE] Brak okien walidacyjnych — używam 10% danych treningowych.")
        n_tr = len(train_data['scg_final'])
        n_val = max(1, int(0.1 * n_tr))
        val_data = {k: train_data[k][:n_val] if train_data[k] is not None else None for k in train_data}
        train_data = {k: train_data[k][n_val:] if train_data[k] is not None else None for k in train_data}

    def _make_dataset(d):
        pcg = torch.tensor(d['gcg_final'], dtype=torch.float32)
        scg = torch.tensor(d['scg_final'], dtype=torch.float32)
        ecg = torch.tensor(d['ecg_final'], dtype=torch.float32).unsqueeze(-1)

        # Dodaj wymiar kanalow jesli brakuje
        if pcg.dim() == 2:
            pcg = pcg.unsqueeze(-1)
        if scg.dim() == 2:
            scg = scg.unsqueeze(-1)

        # Filtruj okna zawierajace NaN lub inf (np. z wavelet na zdegenerowanych fragmentach)
        valid_mask = (
            ~torch.isnan(pcg).any(dim=(1, 2)) &
            ~torch.isnan(scg).any(dim=(1, 2)) &
            ~torch.isnan(ecg).any(dim=(1, 2)) &
            ~torch.isinf(pcg).any(dim=(1, 2)) &
            ~torch.isinf(scg).any(dim=(1, 2))
        )
        n_before = len(pcg)
        pcg, scg, ecg = pcg[valid_mask], scg[valid_mask], ecg[valid_mask]
        n_removed = n_before - len(pcg)
        if n_removed > 0:
            print(f"[NaN-filter] Usunieto {n_removed} okien z NaN/inf z {n_before} lacznie.")

        print(f"[DIAG] pcg: {pcg.shape}, scg: {scg.shape}, ecg: {ecg.shape}")
        return TensorDataset(pcg, scg, ecg)

    train_dataset = _make_dataset(train_data)
    val_dataset   = _make_dataset(val_data)
    num_samples = len(train_dataset)
    print(f"\n[SUKCES] Okna treningowe: {num_samples}, walidacyjne: {len(val_dataset)}")

    # DataLoader walidacyjny — bez tasowania, zawsze deterministyczny
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # DataLoader treningowy z seedem per epoka — gwarantuje identyczną kolejność mini-batchy
    # przy wznowieniu. Epoka globalna 51 zawsze dostaje te same batche co przy przebiegu od zera.
    def _train_loader(global_epoch: int) -> TorchDataLoader:
        g = torch.Generator()
        g.manual_seed(SPLIT_SEED + global_epoch)
        return TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)

    # Diagnostyka danych przed treningiem
    sample_pcg, sample_scg, sample_ecg = train_dataset[0]
    print(f"\n[DIAG] Przykładowe okno treningowe #0:")
    print(f"  PCG/GCG: shape={sample_pcg.shape}, mean={sample_pcg.mean():.4f}, std={sample_pcg.std():.4f}")
    print(f"  SCG: shape={sample_scg.shape}, mean={sample_scg.mean():.4f}, std={sample_scg.std():.4f}")
    print(f"  ECG: shape={sample_ecg.shape}, mean={sample_ecg.mean():.4f}, std={sample_ecg.std():.4f}")

    # Określenie wymiaru wejściowego (3 dla subband, 1 dla standardu)
    input_dim = sample_scg.shape[-1]

    print(f"\nInicjalizacja modelu '{model_name}' z input_dim={input_dim}...")
    from models.architectures import ARCHITECTURE_REGISTRY
    ModelClass = ARCHITECTURE_REGISTRY.get(model_name, ECGReconstructionModel)
    model = ModelClass(input_dim=input_dim).to(device)
    print(f"Model jest na urządzeniu: {next(model.parameters()).device}")
    
    # Zastosowanie bezpieczniejszego współczynnika uczenia (learning rate) w zależności od architektury w celu zapobieżenia eksplozji gradientu (NaN)
    lr = 0.0005 if model_name in ['tcn', 'bilstm_transformer', 'bilstm_transformer_v2'] else 0.002
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # T_max celowo duże (10× epochs, min. 1000) — cosinus nie kończy jednego okresu
    # w typowym treningu. Dzięki temu wznowienie jest idealne: scheduler.step() w epoce 51
    # daje identyczny LR niezależnie od tego, czy trening był 100-epokowy czy 50+50.
    t_max = max(epochs * 10, 1000)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-5)
    criterion = nn.MSELoss()

    weights_path = f"models/{model_name}_{pipeline_name}_best.pth"
    state_path   = f"models/{model_name}_{pipeline_name}_state.pt"
    # Fallback do starej nazwy dla kompatybilności wstecznej
    fallback_path = "models/global_best_ecg_model.pth"

    # Stan treningu — inicjalizacja domyślna (nadpisana przy wznowieniu)
    start_epoch   = 0
    best_corr     = -1.0
    best_val_loss = float('inf')
    es_counter    = 0

    if resume:
        if os.path.exists(state_path):
            # ── Pełne wznowienie: model + optymalizator + scheduler + stan ──────
            try:
                ckpt = torch.load(state_path, map_location=device)
                start_epoch = ckpt['epoch']
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])

                # Odtwórz scheduler z zapisanym T_max (może różnić się od bieżącej sesji)
                saved_t_max = ckpt.get('t_max', t_max)
                if saved_t_max != t_max:
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=saved_t_max, eta_min=1e-5)
                    t_max = saved_t_max
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])

                best_corr     = ckpt.get('best_corr',     -1.0)
                best_val_loss = ckpt.get('best_val_loss', float('inf'))
                es_counter    = ckpt.get('es_counter',    0)

                print(f"  [RESUME] Pełny stan treningu wczytany z '{state_path}'")
                print(f"  Epoki ukończone: {start_epoch}  |  LR: {optimizer.param_groups[0]['lr']:.6f}  |  "
                      f"Najlepsza korelacja: {best_corr:.4f}")
            except Exception as e:
                print(f"  [OSTRZEŻENIE] Nie udało się wczytać stanu: {e}")
                print("  Próbuję wczytać tylko wagi (stary format)...")
                target_path = weights_path if os.path.exists(weights_path) else (
                    fallback_path if os.path.exists(fallback_path) else None)
                if target_path:
                    try:
                        model.load_state_dict(torch.load(target_path, map_location=device))
                        _, val_corr = validate(model, val_loader, criterion)
                        best_corr = val_corr
                        print(f"  Wagi wczytane z '{target_path}', korelacja: {best_corr:.4f}")
                    except Exception as e2:
                        print(f"  [OSTRZEŻENIE] Nie udało się wczytać wag: {e2}. Trening od zera.")
        else:
            # ── Brak state.pt — stary format (tylko wagi) ────────────────────
            target_path = weights_path if os.path.exists(weights_path) else (
                fallback_path if os.path.exists(fallback_path) else None)
            if target_path:
                try:
                    model.load_state_dict(torch.load(target_path, map_location=device))
                    print(f"  [RESUME] Wagi wczytane z '{target_path}' (brak state.pt — stary format).")
                    print("  Obliczanie korelacji na walidacji...")
                    _, val_corr = validate(model, val_loader, criterion)
                    best_corr = val_corr
                    print(f"  Korelacja walidacyjna: {best_corr:.4f}")
                except Exception as e:
                    print(f"  [OSTRZEŻENIE] Nie udało się wczytać wag: {e}. Trening od zera.")
            else:
                print(f"  [RESUME] Brak pliku stanu ani wag — trening od zera.")

    print("\n[TRENING] Rozpoczęcie iteracji...")
    if patience is not None:
        print(f"[TRENING] Early stopping aktywny: patience={patience} epok bez poprawy val_loss.")
    training_results_dir = os.path.join("results", "training")
    os.makedirs("models", exist_ok=True)
    os.makedirs(training_results_dir, exist_ok=True)

    history = []
    total_epochs = start_epoch + epochs   # globalna liczba epok po zakończeniu tej sesji

    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss = train_epoch(model, _train_loader(epoch), optimizer, criterion, augment=augment)
        val_loss, val_corr = validate(model, val_loader, criterion)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{total_epochs:02d} -> Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val Corr: {val_corr:.4f} | LR: {lr_now:.6f}")

        # Record epoch statistics
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_corr': val_corr,
            'lr': lr_now
        })

        # Zapis najlepszego modelu (oparty o korelację)
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), weights_path)
            print(f"  [+] Zapisano nowy '{weights_path}' (Korelacja: {best_corr:.4f})")

        # Early stopping – śledzi val_loss (bardziej stabilne niż korelacja)
        if patience is not None:
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= patience:
                    print(f"  [Early Stop] Brak poprawy val_loss przez {patience} epok. "
                          f"Zatrzymanie na epoce {epoch+1}.")
                    # Zapisz stan przed wyjściem
                    torch.save({
                        'epoch':                epoch + 1,
                        'model_state_dict':     model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_corr':            best_corr,
                        'best_val_loss':        best_val_loss,
                        'es_counter':           es_counter,
                        't_max':                t_max,
                    }, state_path)
                    break

        # Zapis checkpointu co 50 epok (globalna numeracja)
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f"models/{model_name}_{pipeline_name}_checkpoint_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [Checkpoint] Zapisano: {checkpoint_path}")

        # Zapisz pełny stan treningu po każdej epoce
        # Dzięki temu wznowienie (resume) jest zawsze spójne matematycznie:
        #   kolejność mini-batchy, LR i stan Adam są dokładnie takie, jak gdyby
        #   trening przebiegał bez przerwy.
        torch.save({
            'epoch':                epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_corr':            best_corr,
            'best_val_loss':        best_val_loss,
            'es_counter':           es_counter,
            't_max':                t_max,
        }, state_path)

    print(f"\n[Trening zakończony] Najlepsza korelacja walidacyjna: {best_corr:.4f}")
    return model

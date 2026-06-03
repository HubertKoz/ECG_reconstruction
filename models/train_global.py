import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from data_loader import DataLoader as ECGDataLoader
from preprocessing import Preprocessor
from .model import ECGReconstructionModel

import warnings
warnings.filterwarnings('ignore')

# Konfiguracja urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    valid_batches = 0
    for batch_idx, (pcg, scg, target_ecg) in enumerate(train_loader):
        pcg, scg, target_ecg = pcg.to(device), scg.to(device), target_ecg.to(device)

        # Pomijaj batch jesli wejscie zawiera NaN (np. z wavelet na zdegenerowanych oknach)
        if torch.isnan(pcg).any() or torch.isnan(scg).any() or torch.isnan(target_ecg).any():
            continue

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
    pre = Preprocessor(fs=256)
    
    # Słownik potoków do dynamicznego wyboru
    from preprocessing.pipelines import kaisti_pipeline, advanced_filtering_pipeline, aggregate_and_balance_datasets
    from preprocessing.alternative_pipelines import ALTERNATIVE_PIPELINES
    
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
        
        # Łączymy wszystkie ramki danych w jedną listę do globalnego balansowania rekordów
        all_dfs = []
        for ds_name in all_dfs_dict:
            all_dfs.extend(all_dfs_dict[ds_name])
        
        if not all_dfs:
            print("[BŁĄD] Nie znaleziono żadnych danych do treningu.")
            return

        print(f"\n========== Przetwarzanie i balansowanie danych ({len(all_dfs)} rekordów) ==========")

        # Stratified split na poziomie rekordów — zapobiega data leakage między pacjentami
        random.seed(42)
        indices = list(range(len(all_dfs)))
        random.shuffle(indices)
        n_val_records = max(1, int(0.2 * len(all_dfs)))
        val_indices = set(indices[:n_val_records])
        train_dfs = [all_dfs[i] for i in indices if i not in val_indices]
        val_dfs   = [all_dfs[i] for i in val_indices]
        print(f"  Rekordy treningowe: {len(train_dfs)}, walidacyjne: {len(val_dfs)}")

        train_data = aggregate_and_balance_datasets(train_dfs, fs=256, pipeline_func=pipeline_fn, seq_len=250)
        val_data   = aggregate_and_balance_datasets(val_dfs,   fs=256, pipeline_func=pipeline_fn, seq_len=250)
        
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

    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = TorchDataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

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
    
    # Używamy bezpieczniejszej szybkości uczenia w zależności od architektury, aby zapobiec eksplozji gradientu (NaN)
    lr = 0.0005 if model_name in ['tcn', 'bilstm_transformer'] else 0.002
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_corr = -1.0
    weights_path = f"models/{model_name}_{pipeline_name}_best.pth"
    # Fallback do starej nazwy dla kompatybilności wstecznej
    fallback_path = "models/global_best_ecg_model.pth"
    
    if resume:
        target_path = weights_path if os.path.exists(weights_path) else (fallback_path if os.path.exists(fallback_path) else None)
        if target_path:
            try:
                model.load_state_dict(torch.load(target_path, map_location=device))
                print(f"  [RESUME] Pomyślnie wczytano dotychczasowe wagi z '{target_path}'!")
                print("  Obliczanie początkowej korelacji na walidacji...")
                val_loss, val_corr = validate(model, val_loader, criterion)
                best_corr = val_corr
                print(f"  Początkowa korelacja walidacyjna: {best_corr:.4f}")
            except Exception as e:
                print(f"  [OSTRZEŻENIE] Nie udało się wczytać wag (niezgodność wymiarów): {e}")
                print("  Trening rozpocznie się od nowa.")
        else:
            print(f"  [RESUME] Brak pliku wag — trening rozpocznie się od nowa.")

    print("\n[TRENING] Rozpoczęcie iteracji...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    history = []

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_corr = validate(model, val_loader, criterion)
        scheduler.step()
        
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{epochs:02d} -> Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val Corr: {val_corr:.4f} | LR: {lr_now:.6f}")
        
        # Record epoch statistics
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_corr': val_corr,
            'lr': lr_now
        })

        # Zapis najlepszego modelu
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), weights_path)
            print(f"  [+] Zapisano nowy '{weights_path}' (Korelacja: {best_corr:.4f})")
            
        # Zapis checkpointu co 50 epok
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f"models/{model_name}_{pipeline_name}_checkpoint_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [Checkpoint] Zapisano stan epoki {epoch+1} w '{checkpoint_path}'")

    final_path = f"models/{model_name}_{pipeline_name}_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n[ZAKOŃCZONO] Trening przebiegł pomyślnie. Zapisano {final_path}")

    # Save training history to CSV
    try:
        import pandas as pd
        history_df = pd.DataFrame(history)
        history_csv_path = f"results/{model_name}_{pipeline_name}_history.csv"
        history_df.to_csv(history_csv_path, index=False)
        print(f"  [+] Zapisano historię treningu do '{history_csv_path}'")
    except Exception as e:
        print(f"  [OSTRZEŻENIE] Nie udało się zapisać historii do pliku CSV: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trenowanie modelu ciągłej rekonstrukcji EKG")
    parser.add_argument('--model',      type=str, default='bilstm_transformer', help='Architektura modelu')
    parser.add_argument('--pipeline',   type=str, default='kaisti',             help='Potok preprocessingu')
    parser.add_argument('--epochs',     type=int, default=50,                  help='Liczba epok')
    parser.add_argument('--batch_size', type=int, default=32,                  help='Wielkość batcha')
    parser.add_argument('--new',        action='store_true',                   help='Rozpocznij trening od nowa (domyślnie wznawia z najlepszych wag)')
    parser.add_argument('--preprocess', action='store_true',                   help='Wymuś preprocessing danych i zaktualizuj cache')
    args = parser.parse_args()
    
    main(
        model_name=args.model,
        pipeline_name=args.pipeline,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=(not args.new),
        force_preprocess=args.preprocess
    )

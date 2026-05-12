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
    for batch_idx, (pcg, scg, target_ecg) in enumerate(train_loader):
        pcg, scg, target_ecg = pcg.to(device), scg.to(device), target_ecg.to(device)
        
        optimizer.zero_grad()
        output_ecg = model(pcg, scg)
        loss = criterion(output_ecg, target_ecg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

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

def main():
    loader = ECGDataLoader()
    pre = Preprocessor(fs=256)
    
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

    train_data = pre.aggregate_and_balance(train_dfs, seq_len=250)
    val_data   = pre.aggregate_and_balance(val_dfs,   seq_len=250)

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
        pcg = torch.tensor(d['gcg_final'], dtype=torch.float32).unsqueeze(-1)
        scg = torch.tensor(d['scg_final'], dtype=torch.float32).unsqueeze(-1)
        ecg = torch.tensor(d['ecg_final'], dtype=torch.float32).unsqueeze(-1)
        return TensorDataset(pcg, scg, ecg)

    train_dataset = _make_dataset(train_data)
    val_dataset   = _make_dataset(val_data)
    num_samples = len(train_dataset)
    print(f"\n[SUKCES] Okna treningowe: {num_samples}, walidacyjne: {len(val_dataset)}")

    batch_size = 32
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = TorchDataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # Diagnostyka danych przed treningiem
    sample_pcg, sample_scg, sample_ecg = train_dataset[0]
    print(f"\n[DIAG] Przykładowe okno treningowe #0:")
    print(f"  PCG: mean={sample_pcg.mean():.4f}, std={sample_pcg.std():.4f}, min={sample_pcg.min():.4f}, max={sample_pcg.max():.4f}")
    print(f"  SCG: mean={sample_scg.mean():.4f}, std={sample_scg.std():.4f}, min={sample_scg.min():.4f}, max={sample_scg.max():.4f}")
    print(f"  ECG: mean={sample_ecg.mean():.4f}, std={sample_ecg.std():.4f}, min={sample_ecg.min():.4f}, max={sample_ecg.max():.4f}")

    print("\nInicjalizacja modelu BiLSTM + Transformer...")
    model = ECGReconstructionModel().to(device)
    print(f"Model jest na urządzeniu: {next(model.parameters()).device}")
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    criterion = nn.MSELoss()

    num_epochs = 50
    best_corr = -1.0
    print("\n[TRENING] Rozpoczęcie iteracji...")
    os.makedirs("models", exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_corr = validate(model, val_loader, criterion)
        scheduler.step()
        
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} -> Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | Val Corr: {val_corr:.4f} | LR: {lr_now:.6f}")
        
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), "models/global_best_ecg_model.pth")
            print(f"  [+] Zapisano nowy 'models/global_best_ecg_model.pth' (Korelacja: {best_corr:.4f})")

    torch.save(model.state_dict(), "models/global_final_ecg_model.pth")
    print("\n[ZAKOŃCZONO] Trening przebiegł pomyślnie. Zapisano models/global_final_ecg_model.pth")

if __name__ == "__main__":
    main()

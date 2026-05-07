import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

# Zależności z Twojego projektu
from data_loader import DataLoader as ECGDataLoader
from preprocessing import Preprocessor
from .model import ECGReconstructionModel

# Ograniczenia ostrzeżeń
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
    
    # Listowanie wszystkich rekordów dla każdego datasetu
    records_ieee = loader.list_ieee()
    records_zenodo = loader.list_zenodo()
    
    valid_scg_all, valid_pcg_all, valid_ecg_all = [], [], []
    
    seq_len = 250
    fs = 256
    
    def normalize_window(x):
        """Per-window z-score normalizacja (zamiast globalnej z zerami)."""
        std = np.std(x)
        if std < 1e-8:
            return None  # okno jest stałe (same zera lub brak zmienności) - odrzuć
        return (x - np.mean(x)) / std
    
    # Funkcja pomocnicza do przetwarzania i zrzucania okien do wspólnej puli
    def process_and_aggregate_df(signals_df):
        if signals_df is None or signals_df.empty:
            return
            
        try:
            results = pre.process_pipeline(signals_df)
            # Używamy przefiltrowanych (ale NIE zróżniczkowanych) sygnałów
            scg_channel = results['scg_f']
            pcg_channel = results['gcg_f']
            ecg_channel = results['ecg_kaisti']  # EKG z z-score jest OK (nie ma zerowania)
            clean_mask = results['clean_mask']
            
            n_samples_epoch = int(results['epoch_sec'] * fs)
            n_samples_total = len(scg_channel)
            n_windows = n_samples_total // seq_len
            
            for i in range(n_windows):
                start = i * seq_len
                end = start + seq_len
                
                epoch_start = start // n_samples_epoch
                epoch_end = (end - 1) // n_samples_epoch
                
                # Zabezpieczenie przed epokami poza maską i odrzucenie zepsutych okien szumowych
                if epoch_start < len(clean_mask) and epoch_end < len(clean_mask):
                    if clean_mask[epoch_start] and clean_mask[epoch_end]:
                        # Per-window normalizacja zamiast globalnej
                        scg_win = normalize_window(scg_channel[start:end])
                        pcg_win = normalize_window(pcg_channel[start:end])
                        ecg_win = normalize_window(ecg_channel[start:end])
                        
                        # Odrzucamy okna, w których którykolwiek kanał jest stały
                        if scg_win is not None and pcg_win is not None and ecg_win is not None:
                            valid_scg_all.append(scg_win)
                            valid_pcg_all.append(pcg_win)
                            valid_ecg_all.append(ecg_win)
        except Exception as e:
            print(f"Błąd podczas przetwarzania potoku Kaisti: {e}")

    print("========== Rozpoczęcie ładowania i przetwarzania danych IEEE ==========")
    for rec in records_ieee:
        print(f" -> Rekord IEEE: {rec}")
        df = loader.load_ieee(record=rec, format=True)
        if df is not None:
            process_and_aggregate_df(df)

    print("\n========== Rozpoczęcie ładowania i przetwarzania danych Zenodo ==========")
    for rec in records_zenodo:
        print(f" -> Rekord Zenodo: {rec}")
        df = loader.load_zenodo(record=rec, format=True)
        if df is not None:
            process_and_aggregate_df(df)

    if len(valid_scg_all) == 0:
        print("\n[BŁĄD] Nie zebrano żadnych prawidłowych okien do treningu. Upewnij się, że struktura plików jest prawidłowa.")
        return

    # Zamiana ustrukturyzowanych list na wielkie Zespolone Tensory
    real_scg = torch.tensor(np.array(valid_scg_all), dtype=torch.float32).unsqueeze(-1)
    real_pcg = torch.tensor(np.array(valid_pcg_all), dtype=torch.float32).unsqueeze(-1)
    real_ecg = torch.tensor(np.array(valid_ecg_all), dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(real_pcg, real_scg, real_ecg)
    num_samples = len(dataset)
    print(f"\n[SUKCES] Przygotowano łącznie: {num_samples} fragmentów (okien) z wszystkich zbiorów do treningu!")
    
    # Poprawny podział wymieszanych danych ze wszystkich zbiorów pacjentów
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Diagnostyka danych przed treningiem
    sample_pcg, sample_scg, sample_ecg = dataset[0]
    print(f"\n[DIAG] Przykładowe okno #0:")
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

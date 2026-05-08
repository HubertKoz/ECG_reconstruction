import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

# Zależności z projektu
from data_loader import DataLoader as ECGDataLoader
from preprocessing import Preprocessor
from .model_hr import HRVBeatDetectionModel
from utils_peaks import extract_r_peaks, generate_beat_mask

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Klasa funkcji straty rozwiązująca problem niezbalansowania (dużo tła, mało pików)
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1):
        # Wejścia są już po funkcji Sigmoid w modelu!
        bce_loss = self.bce(inputs, targets)
        
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)  
        
        return bce_loss + dice_loss

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for pcg, scg, target_mask in train_loader:
        pcg, scg, target_mask = pcg.to(device), scg.to(device), target_mask.to(device)
        
        optimizer.zero_grad()
        output_mask = model(pcg, scg)
        loss = criterion(output_mask, target_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for pcg, scg, target_mask in val_loader:
            pcg, scg, target_mask = pcg.to(device), scg.to(device), target_mask.to(device)
            output_mask = model(pcg, scg)
            
            loss = criterion(output_mask, target_mask)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def main():
    loader = ECGDataLoader()
    pre = Preprocessor(fs=256)
    
    # Wszystkie dostępne rekordy
    records_ieee = loader.list_ieee()
    records_zenodo = loader.list_zenodo()
    
    valid_scg_all, valid_pcg_all, valid_target_masks = [], [], []
    
    # Zwiększymy okno do około 4 sekund (1000 próbek / 256 Hz), aby złapać kilka bić serca w oknie
    seq_len = 1000
    fs = 256
    
    def process_and_aggregate_df(signals_df):
        if signals_df is None or signals_df.empty:
            return
            
        try:
            results = pre.process_pipeline(signals_df)
            scg_channel = results['scg_final']
            pcg_channel = results['gcg_final']  
            ecg_channel = results['ecg_final']
            clean_mask = results['clean_mask']
            
            n_samples_epoch = int(results['epoch_sec'] * fs)
            n_samples_total = len(scg_channel)
            n_windows = n_samples_total // seq_len
            
            for i in range(n_windows):
                start = i * seq_len
                end = start + seq_len
                
                epoch_start = start // n_samples_epoch
                epoch_end = (end - 1) // n_samples_epoch
                
                # Tylko czyste epoki
                if epoch_start < len(clean_mask) and epoch_end < len(clean_mask):
                    if clean_mask[epoch_start] and clean_mask[epoch_end]:
                        
                        # --- GROUND TRUTH EXTRACTION ---
                        # Zamiast wrzucać EKG, ekstrahujemy markery uderzeń
                        window_ecg = ecg_channel[start:end]
                        peaks = extract_r_peaks(window_ecg, fs=fs)
                        
                        # Tworzymy maskę Gaussa z uderzeniami jako docelowe prawdopodobieństwa
                        mask = generate_beat_mask(seq_len, peaks, sigma=2.0)
                        
                        valid_scg_all.append(scg_channel[start:end])
                        valid_pcg_all.append(pcg_channel[start:end])
                        valid_target_masks.append(mask)
                        
        except Exception as e:
            print(f"Błąd podczas przetwarzania potoku Kaisti (generacja peaków z EKG): {e}")

    print("========== Generacja połączonego zbioru Ground Truth ==========")
    for rec in records_ieee:
        print(f" -> Rekord IEEE: {rec}")
        df = loader.load_ieee(record=rec, format=True)
        process_and_aggregate_df(df)

    for rec in records_zenodo:
        print(f" -> Rekord Zenodo: {rec}")
        df = loader.load_zenodo(record=rec, format=True)
        process_and_aggregate_df(df)

    if len(valid_scg_all) == 0:
        print("\n[BŁĄD] Nie zebrano żadnych prawidłowych okien do treningu.")
        return

    # Zamiana ustrukturyzowanych list na wielkie Tensory
    real_scg = torch.tensor(np.array(valid_scg_all), dtype=torch.float32).unsqueeze(-1)
    real_pcg = torch.tensor(np.array(valid_pcg_all), dtype=torch.float32).unsqueeze(-1)
    
    # Maski to również tensor float [Batch, SeqLen, 1]
    target_masks = torch.tensor(np.array(valid_target_masks), dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(real_pcg, real_scg, target_masks)
    num_samples = len(dataset)
    print(f"\n[SUKCES] Wyodrębniono indeksy HR i przygotowano {num_samples} fragmentów x {seq_len} próbek!")
    
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 16
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("\nInicjalizacja HRV Beat Detection Model...")
    model = HRVBeatDetectionModel(input_dim=1, hidden_dim=64, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = DiceBCELoss()

    num_epochs = 50
    best_loss = float('inf')
    
    print("\n[TRENING] Rozpoczęcie iteracji...")
    os.makedirs("models", exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "models/global_best_hr_model.pth")
            print(f"  [+] Zapisano nowy 'models/global_best_hr_model.pth' (Loss: {best_loss:.4f})")

    torch.save(model.state_dict(), "models/global_final_hr_model.pth")
    print("\n[ZAKOŃCZONO] Sieć HR Model gotowa. Zapisano models/global_final_hr_model.pth")

if __name__ == "__main__":
    main()

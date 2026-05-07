import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

# Zakładamy, że moduły data_loader, preprocessing i model są w tym samym katalogu.
from data_loader import DataLoader as ECGDataLoader
from preprocessing import Preprocessor
from models.model import ECGReconstructionModel

def evaluate_and_plot(model_path='models/best_ecg_model.pth', record='CP-01', num_samples=3, base_data_dir='./data'):
    """
    Funkcja ładująca przykładowe dane (okna) i używająca wytrenowanego modelu
    do rekonstrukcji sygnału EKG na bazie SCG i GCG.
    Na koniec rysuje wykresy dla podanej liczby próbek.
    """
    print(f"[1] Konfiguracja środowiska...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inicjalizacja modelu i załadowanie wag
    model = ECGReconstructionModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f" -> Załadowano wagi modelu z: {model_path}")
    except FileNotFoundError:
        print(f" -> Błąd: Nie znaleziono pliku {model_path}! Wytrenuj model najpierw.")
        return
    model.eval()

    print(f"[2] Pobieranie danych...")
    loader = ECGDataLoader(base_data_dir=base_data_dir) 
    pre = Preprocessor(fs=256)
    
    signals_df = loader.load_zenodo(record=record)
    if signals_df is None:
        print(" -> Brak danych Zenodo. Próba ładowania testowych IEEE...")
        signals_df = loader.load_ieee(record='sub_1', format=True)

    if signals_df is None:
        print(" -> Błąd: Brak danych do ewaluacji.")
        return

    print(f"[3] Przetwarzanie potokiem Kaisti...")
    results = pre.process_pipeline(signals_df)
    
    # Ekstrakcja znormalizowanych sygnałów
    scg_channel = results['scg_kaisti']
    pcg_channel = results['gcg_kaisti'] # W modelu PCG zajmuje miejsce GCG
    ecg_channel = results['ecg_kaisti']
    
    clean_mask = results['clean_mask']
    fs = 256
    seq_len = 250
    n_samples_epoch = int(results['epoch_sec'] * fs)
    n_windows = len(scg_channel) // seq_len
    
    valid_scg, valid_pcg, valid_ecg = [], [], []

    # Selekcja czystych okien
    for i in range(n_windows):
        start = i * seq_len
        end = start + seq_len
        epoch_start = start // n_samples_epoch
        epoch_end = (end - 1) // n_samples_epoch
        if epoch_start < len(clean_mask) and epoch_end < len(clean_mask):
            if clean_mask[epoch_start] and clean_mask[epoch_end]:
                valid_scg.append(scg_channel[start:end])
                valid_pcg.append(pcg_channel[start:end])
                valid_ecg.append(ecg_channel[start:end])

    print(f" -> Czystych okien do weryfikacji: {len(valid_scg)}")
    
    # Wybór `num_samples` losowych / stałych okien (np. z końca, by to na nich przetestować)
    # Bierzemy okna np. ze środka, by wykres był ciekawszy
    idx_start = len(valid_scg) // 2
    
    if len(valid_scg) == 0:
        print("Brak czystych okien.")
        return
        
    for k in range(min(num_samples, len(valid_scg))):
        idx = idx_start + k
        
        # Pojedyncza próbka ma wymiar [1, seq_len, 1] - symulacja batcha 1 elementu
        input_scg = torch.tensor(valid_scg[idx], dtype=torch.float32).view(1, seq_len, 1).to(device)
        input_pcg = torch.tensor(valid_pcg[idx], dtype=torch.float32).view(1, seq_len, 1).to(device)
        target_ecg = valid_ecg[idx]
        
        print(f"[4] Predykcja dla Próbki #{k+1}...")
        with torch.no_grad():
            output_ecg = model(input_pcg, input_scg)
            # Przywrócenie do formatu Numpy
            pred_ecg = output_ecg.cpu().squeeze().numpy()
            
        # Obliczenie lokalnej korelacji
        corr = np.corrcoef(pred_ecg, target_ecg)[0, 1]
        
        print(f" -> Korelacja dla próbki: {corr:.4f}")
        
        # Wizualizacja 
        plt.figure(figsize=(15, 8))
        time_axis = np.arange(seq_len) / fs
        
        # Subplot 1: Wejścia (SCG i GCG)
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, valid_scg[idx], label="Wejście: SCG (Z-score)", color='#1f77b4', linewidth=1.5)
        plt.plot(time_axis, valid_pcg[idx], label="Wejście: GCG (Z-score)", color='#ff7f0e', linewidth=1.5, alpha=0.8)
        plt.title(f"Sygnały Mechaniczne (Wejście do Modelu) - Wynik Korelacji z EKG: {corr:.4f}")
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda znormalizowana")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Subplot 2: Przewidziane vs Prawdziwe EKG
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, target_ecg, label="Referencyjne EKG (Ground Truth)", color='black', linewidth=2.0)
        plt.plot(time_axis, pred_ecg, label="Zrekonstruowane EKG (Predykcja Modelu)", color='red', linestyle='--', linewidth=2.0)
        plt.title("Rekonstrukcja fali kardiologicznej (Wyjście z Modelu)")
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda znormalizowana")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    evaluate_and_plot(model_path='best_ecg_model.pth', num_samples=3)

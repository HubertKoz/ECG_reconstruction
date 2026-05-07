import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from data_loader import DataLoader as ECGDataLoader
from preprocessing import Preprocessor
from models.model_hr import HRVBeatDetectionModel
from utils_peaks import extract_r_peaks, refine_peak_parabolic

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_hrv_indices(r_peaks, fs=256):
    """
    Oblicza podstawowe indeksy HRV na podstawie lokalizacji pików.
    """
    if len(r_peaks) < 2:
        return {'HeartRate': 0, 'SDNN': 0, 'RMSSD': 0}
        
    rr_intervals_samples = np.diff(r_peaks)
    rr_intervals_ms = (rr_intervals_samples / fs) * 1000.0
    
    # Usuwanie anomalii fizjologicznych (np. błędy rzędu 2 sekund między pobudzeniami)
    valid_rr = rr_intervals_ms[(rr_intervals_ms > 300) & (rr_intervals_ms < 2000)]
    
    if len(valid_rr) < 2:
         return {'HeartRate': 0, 'SDNN': 0, 'RMSSD': 0}
         
    mean_rr = np.mean(valid_rr)
    hr = 60000.0 / mean_rr
    
    sdnn = np.std(valid_rr, ddof=1)
    
    diff_rr = np.diff(valid_rr)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    
    return {
        'HeartRate': np.round(hr, 1),
        'SDNN': np.round(sdnn, 2),
        'RMSSD': np.round(rmssd, 2)
    }

def evaluate_record(record_name='CP-01', dataset='Zenodo', model_path='models/global_best_hr_model.pth', base_data_dir='./data'):
    loader = ECGDataLoader(base_data_dir=base_data_dir)
    pre = Preprocessor(fs=256)
    
    if dataset.lower() == 'zenodo':
        df = loader.load_zenodo(record=record_name, format=True)
    else:
        df = loader.load_ieee(record=record_name, format=True)
        
    if df is None or df.empty:
        print(f"Nie udało się załadować rekordu {record_name}")
        return
        
    results = pre.process_pipeline(df)
    
    scg_full = results['scg_kaisti']
    pcg_full = results['gcg_kaisti']
    ecg_full = results['ecg_kaisti']
    
    seq_len = 1000
    n_windows = len(scg_full) // seq_len

    # Przydatne stałe
    fs = 256
    epoch_sec = results.get('epoch_sec', 10)
    n_samples_epoch = int(epoch_sec * fs)
    clean_mask = results.get('clean_mask', None)

    # 1. Automatyczne szukanie pierwszego czystego segmentu do wizualizacji
    visualize_start_sec = 0
    if clean_mask is not None:
        clean_indices = np.where(clean_mask)[0]
        if len(clean_indices) > 0:
            visualize_start_sec = clean_indices[0] * epoch_sec
            print(f" -> Automatycznie wybrano segment do wizualizacji od {visualize_start_sec}s (pierwsza czysta epoka).")
        else:
            print(" -> OSTRZEŻENIE: Brak czystych danych (clean_mask jest pusta).")

    # Pobieramy Ground Truth EKG dla całego sygnału
    gt_peaks = extract_r_peaks(ecg_full, fs=fs)
    
    # Filtrowanie GT Peaks: bierzemy tylko te z "czystych" epok do raportu summary
    if clean_mask is not None:
        filtered_gt_peaks = []
        for p in gt_peaks:
            ep_idx = int(p // n_samples_epoch)
            if ep_idx < len(clean_mask) and clean_mask[ep_idx]:
                filtered_gt_peaks.append(p)
        gt_peaks_for_stats = np.array(filtered_gt_peaks)
    else:
        gt_peaks_for_stats = gt_peaks

    # Inicjalizacja modelu
    model = HRVBeatDetectionModel(input_dim=1, hidden_dim=64, num_layers=2)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f" -> Załadowano wagi modelu HR z: {model_path}")
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {model_path}. Upewnij się, że wagi są obecne.")
        return

    predicted_peaks_global = []

    # Iteracja po oknach
    for i in range(n_windows):
        start = i * seq_len
        end = start + seq_len
        
        # Sprawdzanie, czy okno jest w "czystym" obszarze
        if clean_mask is not None:
            ep_start = start // n_samples_epoch
            ep_end = (end - 1) // n_samples_epoch
            # Okno musi być CAŁKOWICIE w obrębie czystych epok
            if ep_start >= len(clean_mask) or ep_end >= len(clean_mask) or not clean_mask[ep_start] or not clean_mask[ep_end]:
                continue

        scg_win = scg_full[start:end]
        pcg_win = pcg_full[start:end]
        
        # Diagnostyka przed normalizacją (dla pierwszego przetwarzanego okna)
        if len(predicted_peaks_global) == 0:
            print(f"[DEBUG] Pierwsze poprawne okno ({i}) SCG raw: max={np.max(scg_win):.2e}, min={np.min(scg_win):.2e}, std={np.std(scg_win):.2e}")

        # Normalizacja Z-score dla każdego okna
        scg_win = (scg_win - np.mean(scg_win)) / (np.std(scg_win) + 1e-9)
        pcg_win = (pcg_win - np.mean(pcg_win)) / (np.std(pcg_win) + 1e-9)

        # Diagnostyka po normalizacji
        if len(predicted_peaks_global) == 0:
            print(f"[DEBUG] Pierwsze poprawne okno ({i}) SCG norm: max={np.max(scg_win):.4f}, min={np.min(scg_win):.4f}, std={np.std(scg_win):.4f}")
        
        # Pomiń okna, które mimo wszystko są płaskie
        if np.std(scg_full[start:end]) < 1e-7:
            continue

        t_scg = torch.tensor(scg_win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        t_pcg = torch.tensor(pcg_win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        
        with torch.no_grad():
            pred_mask = model(t_pcg, t_scg)
            pred_mask_np = pred_mask.cpu().squeeze().numpy()
            
        # Dynamiczny próg detekcji
        max_pred = np.max(pred_mask_np)
        threshold = 0.5 * max_pred if max_pred > 0.1 else 0.3
        peaks_local, _ = find_peaks(pred_mask_np, height=threshold, distance=int(0.3 * fs))
        
        # Sub-próbkowa rafinacja pików (redukcja jitteru)
        peaks_local_refined = [refine_peak_parabolic(pred_mask_np, p) for p in peaks_local]
        
        # Mapa na wektory globalne
        predicted_peaks_global.extend(np.array(peaks_local_refined) + start)
        
    predicted_peaks_global = np.array(predicted_peaks_global)
    
    # Kalkulacja wskaźników (na przefiltrowanych GT peaks)
    hrv_gt = calculate_hrv_indices(gt_peaks_for_stats, fs=fs)
    hrv_pred = calculate_hrv_indices(predicted_peaks_global, fs=fs)
    
    print(f"\n======== RAPORT HRV: {record_name} ({dataset}) ========")
    print(f"{'Metryka':<15} | {'ECG (Filtered GT)':<20} | {'SCG+GCG (Model)':<20} | {'Różnica'}")
    print("-" * 75)
    for key in hrv_gt.keys():
        diff = np.abs(hrv_gt[key] - hrv_pred[key])
        print(f"{key:<15} | {hrv_gt[key]:<20} | {hrv_pred[key]:<20} | {diff:.2f}")

    # Rysowanie wycinka dla wizualizacji
    visualize_start = int(visualize_start_sec * fs)
    visualize_end = int(visualize_start + 10 * fs) # 10 sekund od startu
    
    plt.figure(figsize=(14, 10))
    t = np.arange(visualize_start, visualize_end) / fs
    
    plt.subplot(3, 1, 1)
    plt.plot(t, ecg_full[visualize_start:visualize_end], color='red', label='Referencyjne EKG')
    
    # Markery w wyświetlanym przedziale GT
    disp_gt_peaks = gt_peaks[(gt_peaks >= visualize_start) & (gt_peaks < visualize_end)]
    plt.plot(disp_gt_peaks / 256.0, ecg_full[disp_gt_peaks], 'rx', markersize=10, label='GT R-Peaks')
    plt.title('Referencja (Ground Truth)')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    # Skalowanie wizualizacji: Rysujemy sygnał SCG znormalizowany dla lepszej widoczności pików
    scg_plot_segment = scg_full[visualize_start:visualize_end]
    scg_plot_norm = (scg_plot_segment - np.mean(scg_plot_segment)) / (np.std(scg_plot_segment) + 1e-9)
    plt.plot(t, scg_plot_norm, color='blue', alpha=0.6, label='Sygnał SCG (Z-normalized)')
    
    # Markery w wyświetlanym przedziale PRED (nanoszone na znormalizowany sygnał)
    disp_pred_peaks = predicted_peaks_global[(predicted_peaks_global >= visualize_start) & (predicted_peaks_global < visualize_end)]
    # Mapowanie na lokalne indeksy segmentu wykresu (używamy floor dla dostępu do tablicy)
    local_peaks_idx = (disp_pred_peaks - visualize_start).astype(int)
    plt.plot(disp_pred_peaks / 256.0, scg_plot_norm[local_peaks_idx], 'gx', markersize=10, label='Predykcje Sieci (HR)')
    plt.title('Detekcja mechaniczna modelu (Sygnał znormalizowany)')
    plt.legend()

    plt.subplot(3, 1, 3)
    # Wizualizacja czasowa: GT vs Predykcja na tle EKG
    plt.plot(t, ecg_full[visualize_start:visualize_end], color='gray', alpha=0.3, label='Referencyjne EKG')
    plt.vlines(disp_gt_peaks / 256.0, -2, 2, colors='r', linestyles='--', label='GT Peaks (ECG)')
    plt.vlines(disp_pred_peaks / 256.0, -2, 2, colors='g', linestyles='-', label='Pred Peaks (Model)')
    plt.title('Synchronizacja czasowa pików: Referencja vs Predykcja')
    plt.xlabel('Czas (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"hrv_eval_{record_name}.png")
    print(f"\nWygenerowano i zapisano plik z wykresem 'hrv_eval_{record_name}.png'.")

if __name__ == "__main__":
    evaluate_record(record_name='CP-01', dataset='Zenodo')

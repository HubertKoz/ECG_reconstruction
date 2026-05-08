import torch
import numpy as np
from scipy.signal import find_peaks

from data_loader import DataLoader as ECGDataLoader
from preprocessing import Preprocessor
from models.model import ECGReconstructionModel
from models.model_hr import HRVBeatDetectionModel
from utils_peaks import extract_r_peaks, refine_peak_parabolic
from evaluation.metrics import calculate_hrv_indices, plot_reconstruction, plot_hrv_comparison

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_reconstruction_pipeline(model_path='models/global_best_ecg_model.pth', record='CP-01', num_samples=3, base_data_dir='./data'):
    """
    Pełny proces ewaluacji modelu rekonstrukcji EKG.
    """
    print(f"[1] Konfiguracja środowiska...")
    
    # Inicjalizacja modelu i załadowanie wag
    model = ECGReconstructionModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f" -> Załadowano wagi modelu z: {model_path}")
    except FileNotFoundError:
        print(f" -> Błąd: Nie znaleziono pliku {model_path}!")
        return
    model.eval()

    print(f"[2] Pobieranie danych...")
    loader = ECGDataLoader(base_data_dir=base_data_dir) 
    pre = Preprocessor(fs=256)
    
    signals_df = loader.load_zenodo(record=record)
    if signals_df is None:
        print(f" -> Brak danych Zenodo dla rekordu {record}. Próba ładowania testowych IEEE...")
        signals_df = loader.load_ieee(record='sub_1', format=True)

    if signals_df is None:
        print(f" -> Błąd: Brak danych do ewaluacji dla rekordu {record}.")
        return

    print(f"[3] Przetwarzanie sygnałów...")
    results = pre.process_pipeline(signals_df)
    
    fs = 256
    seq_len = 250
    
    # Wykorzystujemy nową funkcję extract_windows dla spójności
    signals = [results['gcg_final'], results['scg_final'], results['ecg_final']]
    windows = pre.extract_windows(
        signals, 
        fs, 
        seq_len=seq_len, 
        clean_mask=results['clean_mask'], 
        epoch_sec=results['epoch_sec']
    )
    
    # Rozpakowujemy wyniki (kolejność musi odpowiadać liście powyżej)
    valid_pcg, valid_scg, valid_ecg = windows[0], windows[1], windows[2]

    print(f" -> Czystych okien do weryfikacji: {len(valid_scg)}")
    
    if len(valid_scg) == 0:
        print("Brak czystych okien.")
        return
        
    idx_start = len(valid_scg) // 2
    for k in range(min(num_samples, len(valid_scg))):
        idx = idx_start + k
        input_scg = torch.tensor(valid_scg[idx], dtype=torch.float32).view(1, seq_len, 1).to(device)
        input_pcg = torch.tensor(valid_pcg[idx], dtype=torch.float32).view(1, seq_len, 1).to(device)
        target_ecg = valid_ecg[idx]
        
        with torch.no_grad():
            output_ecg = model(input_pcg, input_scg)
            pred_ecg = output_ecg.cpu().squeeze().numpy()
            
        corr = np.corrcoef(pred_ecg, target_ecg)[0, 1]
        print(f" -> Korelacja dla próbki #{k+1}: {corr:.4f}")
        
        time_axis = np.arange(seq_len) / fs
        plot_reconstruction(time_axis, valid_scg[idx], valid_pcg[idx], target_ecg, pred_ecg, corr, k+1)

def evaluate_hrv_pipeline(record_name='CP-01', dataset='Zenodo', model_path='models/global_best_hr_model.pth', base_data_dir='./data'):
    """
    Pełny proces ewaluacji modelu detekcji uderzeń i wskaźników HRV.
    """
    print(f"[1] Inicjalizacja pipeline HRV...")
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
    scg_full = results['scg_final']
    pcg_full = results['gcg_final']
    ecg_full = results['ecg_final']
    
    fs = 256
    seq_len = 1000
    n_windows = len(scg_full) // seq_len
    epoch_sec = results.get('epoch_sec', 10)
    n_samples_epoch = int(epoch_sec * fs)
    clean_mask = results.get('clean_mask', None)

    visualize_start_sec = 0
    if clean_mask is not None:
        clean_indices = np.where(clean_mask)[0]
        if len(clean_indices) > 0:
            visualize_start_sec = clean_indices[0] * epoch_sec
            print(f" -> Wybrano segment do wizualizacji od {visualize_start_sec}s.")

    gt_peaks = extract_r_peaks(ecg_full, fs=fs)
    
    if clean_mask is not None:
        filtered_gt_peaks = [p for p in gt_peaks if (int(p // n_samples_epoch) < len(clean_mask) and clean_mask[int(p // n_samples_epoch)])]
        gt_peaks_for_stats = np.array(filtered_gt_peaks)
    else:
        gt_peaks_for_stats = gt_peaks

    model = HRVBeatDetectionModel(input_dim=1, hidden_dim=64, num_layers=2)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f" -> Załadowano model HR z: {model_path}")
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {model_path}.")
        return

    predicted_peaks_global = []
    for i in range(n_windows):
        start = i * seq_len
        end = start + seq_len
        
        if clean_mask is not None:
            ep_start = start // n_samples_epoch
            ep_end = (end - 1) // n_samples_epoch
            if ep_start >= len(clean_mask) or ep_end >= len(clean_mask) or not clean_mask[ep_start] or not clean_mask[ep_end]:
                continue

        scg_win = scg_full[start:end]
        pcg_win = pcg_full[start:end]
        
        scg_win = (scg_win - np.mean(scg_win)) / (np.std(scg_win) + 1e-9)
        pcg_win = (pcg_win - np.mean(pcg_win)) / (np.std(pcg_win) + 1e-9)

        t_scg = torch.tensor(scg_win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        t_pcg = torch.tensor(pcg_win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        
        with torch.no_grad():
            pred_mask = model(t_pcg, t_scg)
            pred_mask_np = pred_mask.cpu().squeeze().numpy()
            
        max_pred = np.max(pred_mask_np)
        threshold = 0.5 * max_pred if max_pred > 0.1 else 0.3
        peaks_local, _ = find_peaks(pred_mask_np, height=threshold, distance=int(0.3 * fs))
        peaks_local_refined = [refine_peak_parabolic(pred_mask_np, p) for p in peaks_local]
        predicted_peaks_global.extend(np.array(peaks_local_refined) + start)
        
    predicted_peaks_global = np.array(predicted_peaks_global)
    hrv_gt = calculate_hrv_indices(gt_peaks_for_stats, fs=fs)
    hrv_pred = calculate_hrv_indices(predicted_peaks_global, fs=fs)
    
    print(f"\n======== RAPORT HRV: {record_name} ({dataset}) ========")
    print(f"{'Metryka':<15} | {'ECG (Filtered GT)':<20} | {'SCG+GCG (Model)':<20} | {'Różnica'}")
    print("-" * 75)
    for key in hrv_gt.keys():
        diff = np.abs(hrv_gt[key] - hrv_pred[key])
        print(f"{key:<15} | {hrv_gt[key]:<20} | {hrv_pred[key]:<20} | {diff:.2f}")

    visualize_start = int(visualize_start_sec * fs)
    visualize_end = int(visualize_start + 10 * fs)
    t = np.arange(visualize_start, visualize_end) / fs
    
    disp_gt_peaks = gt_peaks[(gt_peaks >= visualize_start) & (gt_peaks < visualize_end)]
    disp_pred_peaks = predicted_peaks_global[(predicted_peaks_global >= visualize_start) & (predicted_peaks_global < visualize_end)]
    
    scg_plot_segment = scg_full[visualize_start:visualize_end]
    scg_plot_norm = (scg_plot_segment - np.mean(scg_plot_segment)) / (np.std(scg_plot_segment) + 1e-9)
    
    plot_hrv_comparison(t, ecg_full[visualize_start:visualize_end], disp_gt_peaks, scg_plot_norm, disp_pred_peaks, record_name, fs=fs)

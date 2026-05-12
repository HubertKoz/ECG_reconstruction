import torch
import numpy as np
from scipy.signal import find_peaks

from data_loader import DataLoader as ECGDataLoader
from preprocessing import Preprocessor
from models.model import ECGReconstructionModel
from models.model_hr import HRVBeatDetectionModel
from utils_peaks import extract_r_peaks, refine_peak_parabolic
from evaluation.metrics import (
    calculate_hrv_indices,
    plot_reconstruction,
    plot_reconstruction_quality,
    plot_hrv_comparison,
    plot_poincare,
    plot_hrv_spectrum,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_reconstruction_pipeline(
    model_path='models/global_best_ecg_model.pth',
    record='CP-01',
    num_samples=3,
    base_data_dir='./data'
):
    """
    Ewaluacja modelu rekonstrukcji EKG: predykcja fali + korelacja Pearsona.
    """
    print("[1] Konfiguracja środowiska...")
    model = ECGReconstructionModel().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  -> Załadowano wagi: {model_path}")
    except FileNotFoundError:
        print(f"  -> Błąd: nie znaleziono {model_path}")
        return
    model.eval()

    print("[2] Pobieranie danych...")
    loader = ECGDataLoader(base_data_dir=base_data_dir)
    pre = Preprocessor(fs=256)

    signals_df = loader.load_zenodo(record=record)
    if signals_df is None:
        print(f"  -> Brak Zenodo dla {record}, próba IEEE...")
        signals_df = loader.load_ieee(record='sub_1', format=True)
    if signals_df is None:
        print(f"  -> Błąd: brak danych dla rekordu {record}.")
        return

    print("[3] Przetwarzanie sygnałów...")
    results = pre.process_pipeline(signals_df)

    fs = 256
    seq_len = 250  # spójne z treningiem w train_global.py

    signals = [results['gcg_final'], results['scg_final'], results['ecg_final']]
    windows = pre.extract_windows(
        signals, fs, seq_len=seq_len,
        clean_mask=results['clean_mask'],
        epoch_sec=results['epoch_sec']
    )
    valid_pcg, valid_scg, valid_ecg = windows[0], windows[1], windows[2]

    print(f"  -> Czystych okien: {len(valid_scg)}")
    if len(valid_scg) == 0:
        print("  Brak czystych okien.")
        return

    idx_start = len(valid_scg) // 2
    correlations = []
    for k in range(min(num_samples, len(valid_scg))):
        idx = idx_start + k
        input_scg = torch.tensor(valid_scg[idx], dtype=torch.float32).view(1, seq_len, 1).to(device)
        input_pcg = torch.tensor(valid_pcg[idx], dtype=torch.float32).view(1, seq_len, 1).to(device)
        target_ecg = valid_ecg[idx]

        with torch.no_grad():
            output_ecg = model(input_pcg, input_scg)
            pred_ecg = output_ecg.cpu().squeeze().numpy()

        corr = np.corrcoef(pred_ecg, target_ecg)[0, 1]
        correlations.append(corr)
        print(f"  -> Korelacja próbka #{k+1}: {corr:.4f}")

        time_axis = np.arange(seq_len) / fs
        plot_reconstruction(time_axis, valid_scg[idx], valid_pcg[idx], target_ecg, pred_ecg, corr, k + 1)
        plot_reconstruction_quality(pred_ecg, target_ecg, time_axis, corr, record_name=record, sample_idx=k + 1)

    print(f"\n  Średnia korelacja: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")


def evaluate_hrv_pipeline(
    record_name='CP-01',
    dataset='Zenodo',
    model_path='models/global_best_hr_model.pth',
    base_data_dir='./data'
):
    """
    Ewaluacja modelu detekcji uderzeń (HRVBeatDetectionModel) i wskaźników HRV.
    """
    print("[1] Inicjalizacja pipeline HRV...")
    loader = ECGDataLoader(base_data_dir=base_data_dir)
    pre = Preprocessor(fs=256)

    df = loader.load_zenodo(record=record_name, format=True) if dataset.lower() == 'zenodo' \
        else loader.load_ieee(record=record_name, format=True)

    if df is None or df.empty:
        print(f"  Nie udało się załadować rekordu {record_name}.")
        return

    results = pre.process_pipeline(df)
    scg_full = results['scg_final']
    pcg_full = results['gcg_final']
    ecg_full = results['ecg_final']

    fs = 256
    seq_len = 1000  # ~4 s — spójne z treningiem w train_hr.py
    n_windows = len(scg_full) // seq_len
    epoch_sec = results.get('epoch_sec', 10)
    n_samples_epoch = int(epoch_sec * fs)
    clean_mask = results.get('clean_mask', None)

    visualize_start_sec = 0
    if clean_mask is not None:
        clean_indices = np.where(clean_mask)[0]
        if len(clean_indices) > 0:
            visualize_start_sec = clean_indices[0] * epoch_sec
            print(f"  -> Segment wizualizacji od {visualize_start_sec:.0f} s.")

    gt_peaks = extract_r_peaks(ecg_full, fs=fs)
    if clean_mask is not None:
        gt_peaks_for_stats = np.array([
            p for p in gt_peaks
            if int(p // n_samples_epoch) < len(clean_mask) and clean_mask[int(p // n_samples_epoch)]
        ])
    else:
        gt_peaks_for_stats = gt_peaks

    model = HRVBeatDetectionModel(input_dim=1, hidden_dim=64, num_layers=2)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"  -> Załadowano model HR: {model_path}")
    except FileNotFoundError:
        print(f"  Błąd: nie znaleziono {model_path}.")
        return

    predicted_peaks_global = []
    for i in range(n_windows):
        start = i * seq_len
        end = start + seq_len

        if clean_mask is not None:
            ep_s = start // n_samples_epoch
            ep_e = (end - 1) // n_samples_epoch
            if ep_s >= len(clean_mask) or ep_e >= len(clean_mask) or \
               not clean_mask[ep_s] or not clean_mask[ep_e]:
                continue

        scg_win = scg_full[start:end]
        pcg_win = pcg_full[start:end]

        # Normalizacja per-okno (spójność z train_hr.py)
        scg_win = (scg_win - np.mean(scg_win)) / (np.std(scg_win) + 1e-9)
        pcg_win = (pcg_win - np.mean(pcg_win)) / (np.std(pcg_win) + 1e-9)

        t_scg = torch.tensor(scg_win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        t_pcg = torch.tensor(pcg_win, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)

        with torch.no_grad():
            pred_mask = model(t_pcg, t_scg)
            pred_mask_np = pred_mask.cpu().squeeze().numpy()

        # Próg adaptacyjny: mean + 2*std (bardziej odporny na skalę sygnału)
        thr = np.mean(pred_mask_np) + 2.0 * np.std(pred_mask_np)
        thr = max(thr, 0.1)
        peaks_local, _ = find_peaks(pred_mask_np, height=thr, distance=int(0.3 * fs))
        peaks_refined = [refine_peak_parabolic(pred_mask_np, p) for p in peaks_local]
        predicted_peaks_global.extend(np.array(peaks_refined) + start)

    predicted_peaks_global = np.array(predicted_peaks_global)

    hrv_gt   = calculate_hrv_indices(gt_peaks_for_stats, fs=fs)
    hrv_pred = calculate_hrv_indices(predicted_peaks_global, fs=fs)

    print(f"\n======== RAPORT HRV: {record_name} ({dataset}) ========")
    fmt_h = f"{'Metryka':<15} | {'GT (EKG)':<14} | {'Model (SCG)':<14} | {'Różnica'}"
    print(fmt_h)
    print("-" * len(fmt_h))
    for key in hrv_gt:
        gt_v   = hrv_gt[key]
        pred_v = hrv_pred[key]
        if np.isnan(gt_v) or np.isnan(pred_v):
            diff_str = "N/A"
        else:
            diff_str = f"{abs(gt_v - pred_v):.2f}"
        print(f"{key:<15} | {str(gt_v):<14} | {str(pred_v):<14} | {diff_str}")

    # Wizualizacja (10-sekundowy fragment)
    visualize_start = int(visualize_start_sec * fs)
    visualize_end   = min(len(ecg_full), visualize_start + int(10 * fs))
    t = np.arange(visualize_start, visualize_end) / fs

    disp_gt_peaks   = gt_peaks[(gt_peaks >= visualize_start) & (gt_peaks < visualize_end)]
    disp_pred_peaks = predicted_peaks_global[
        (predicted_peaks_global >= visualize_start) & (predicted_peaks_global < visualize_end)
    ].astype(int)

    scg_seg = scg_full[visualize_start:visualize_end]
    scg_plot_norm = (scg_seg - np.mean(scg_seg)) / (np.std(scg_seg) + 1e-9)

    plot_hrv_comparison(t, ecg_full[visualize_start:visualize_end],
                        disp_gt_peaks, scg_plot_norm, disp_pred_peaks, record_name, fs=fs)
    plot_poincare(gt_peaks_for_stats, fs=fs, record_name=f"{record_name}_GT")
    plot_poincare(predicted_peaks_global, fs=fs, record_name=f"{record_name}_pred")
    plot_hrv_spectrum(gt_peaks_for_stats, fs=fs, record_name=f"{record_name}_GT")
    plot_hrv_spectrum(predicted_peaks_global, fs=fs, record_name=f"{record_name}_pred")


def evaluate_full_pipeline(
    model_ecg_path='models/global_best_ecg_model.pth',
    record='CP-01',
    dataset='Zenodo',
    base_data_dir='./data'
):
    """
    Pełny pipeline end-to-end:
      1. Wczytanie i preprocessing sygnałów SCG/GCG
      2. Rekonstrukcja EKG przez ECGReconstructionModel
      3. Detekcja pików R ze zrekonstruowanego EKG
      4. Obliczenie indeksów HRV i porównanie z ground truth

    Pozwala ocenić, czy błędy rekonstrukcji wpływają na metryki kliniczne HRV.
    """
    print("=" * 60)
    print(f"  FULL PIPELINE: {record} ({dataset})")
    print("=" * 60)

    loader = ECGDataLoader(base_data_dir=base_data_dir)
    pre = Preprocessor(fs=256)

    df = loader.load_zenodo(record=record, format=True) if dataset.lower() == 'zenodo' \
        else loader.load_ieee(record=record, format=True)
    if df is None or df.empty:
        print(f"  Błąd: brak danych dla {record}.")
        return

    results = pre.process_pipeline(df)
    fs = 256
    seq_len = 250

    signals = [results['gcg_final'], results['scg_final'], results['ecg_final']]
    windows = pre.extract_windows(
        signals, fs, seq_len=seq_len,
        clean_mask=results['clean_mask'],
        epoch_sec=results['epoch_sec']
    )
    valid_pcg, valid_scg, valid_ecg = windows[0], windows[1], windows[2]

    if len(valid_scg) == 0:
        print("  Brak czystych okien.")
        return

    # Ładowanie modelu rekonstrukcji
    model = ECGReconstructionModel().to(device)
    try:
        model.load_state_dict(torch.load(model_ecg_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print(f"  Błąd: nie znaleziono {model_ecg_path}.")
        return

    # Rekonstrukcja EKG dla wszystkich okien i sklejenie w jeden sygnał
    reconstructed_segments = []
    gt_segments = []
    correlations = []

    for idx in range(len(valid_scg)):
        inp_scg = torch.tensor(valid_scg[idx], dtype=torch.float32).view(1, seq_len, 1).to(device)
        inp_pcg = torch.tensor(valid_pcg[idx], dtype=torch.float32).view(1, seq_len, 1).to(device)
        with torch.no_grad():
            pred = model(inp_pcg, inp_scg).cpu().squeeze().numpy()
        reconstructed_segments.append(pred)
        gt_segments.append(valid_ecg[idx])
        corr = np.corrcoef(pred, valid_ecg[idx])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    reconstructed_ecg = np.concatenate(reconstructed_segments)
    gt_ecg_full = np.concatenate(gt_segments)
    mean_corr = float(np.mean(correlations)) if correlations else float('nan')
    print(f"\n  Śr. korelacja Pearsona po wszystkich oknach: {mean_corr:.4f}")

    # Detekcja R-peaks ze zrekonstruowanego EKG
    pred_peaks = extract_r_peaks(reconstructed_ecg, fs=fs)
    gt_peaks   = extract_r_peaks(gt_ecg_full, fs=fs)

    hrv_gt   = calculate_hrv_indices(gt_peaks,   fs=fs)
    hrv_pred = calculate_hrv_indices(pred_peaks, fs=fs)

    print(f"\n  --- Indeksy HRV (GT ECG vs zrekonstruowany ECG) ---")
    for key in hrv_gt:
        gt_v = hrv_gt[key]; pred_v = hrv_pred[key]
        diff_str = "N/A" if (np.isnan(gt_v) or np.isnan(pred_v)) else f"{abs(gt_v - pred_v):.2f}"
        print(f"  {key:<15}: GT={str(gt_v):<10}  Pred={str(pred_v):<10}  Δ={diff_str}")

    # Wykresy
    t_full = np.arange(len(reconstructed_ecg)) / fs
    plot_reconstruction_quality(reconstructed_ecg, gt_ecg_full, t_full,
                                mean_corr, record_name=record, sample_idx=0)
    plot_poincare(pred_peaks, fs=fs, record_name=f"{record}_reconstructed")
    plot_hrv_spectrum(pred_peaks, fs=fs, record_name=f"{record}_reconstructed")

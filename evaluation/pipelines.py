import os
import torch
import numpy as np
from scipy.signal import find_peaks

from config import TARGET_FS, SEQ_LEN
from dataset import DataLoader as ECGDataLoader
from dataset import Preprocessor
from models.model import ECGReconstructionModel
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
    base_data_dir='./data',
    model_name='bilstm_transformer',
    pipeline_name='kaisti'
):
    """
    Ewaluacja modelu rekonstrukcji EKG: predykcja fali + korelacja Pearsona.
    """
    print("[1] Konfiguracja środowiska...")
    
    # Wybór pasma wejściowego na podstawie pipeline
    input_dim = 3 if pipeline_name == 'subband' else 1
    
    from models.architectures import ARCHITECTURE_REGISTRY
    ModelClass = ARCHITECTURE_REGISTRY.get(model_name, ECGReconstructionModel)
    model = ModelClass(input_dim=input_dim).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  -> Załadowano wagi ({model_name}): {model_path}")
    except FileNotFoundError:
        print(f"  -> Błąd: nie znaleziono {model_path}")
        return
    model.eval()

    print("[2] Pobieranie danych...")
    loader = ECGDataLoader(base_data_dir=base_data_dir)
    pre = Preprocessor(fs=TARGET_FS)

    # Auto-detekcja zbioru na podstawie nazwy rekordu
    if record.startswith('sub_'):
        signals_df = loader.load_ieee(record=record, format=True)
    else:
        signals_df = loader.load_zenodo(record=record)
    if signals_df is None:
        print(f"  -> Blad: brak danych dla rekordu {record}.")
        return

    print("[3] Przetwarzanie sygnałów...")
    from pipelines import kaisti_pipeline, advanced_filtering_pipeline
    from pipelines import ALTERNATIVE_PIPELINES

    PIPELINES = {
        'kaisti':   kaisti_pipeline,
        'advanced': advanced_filtering_pipeline,
        **ALTERNATIVE_PIPELINES
    }
    pipeline_fn = PIPELINES.get(pipeline_name, kaisti_pipeline)
    results = pipeline_fn(signals_df, fs=TARGET_FS)

    fs     = TARGET_FS
    seq_len = SEQ_LEN

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

    # Ewaluacja na wszystkich oknach (pelny sygnal), nie na probce num_samples
    correlations = []
    for idx in range(len(valid_scg)):
        input_scg = torch.tensor(valid_scg[idx], dtype=torch.float32).view(1, seq_len, input_dim).to(device)
        input_pcg = torch.tensor(valid_pcg[idx], dtype=torch.float32).view(1, seq_len, input_dim).to(device)
        target_ecg = valid_ecg[idx]

        with torch.no_grad():
            output_ecg = model(input_pcg, input_scg)
            pred_ecg = output_ecg.cpu().squeeze().numpy()

        corr = np.corrcoef(pred_ecg, target_ecg)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    # Wykresy dla pierwszych num_samples okien (do wizualizacji)
    plot_dir = os.path.join("results", f"{model_name}_{pipeline_name}", record)
    os.makedirs(plot_dir, exist_ok=True)
    idx_start = len(valid_scg) // 2
    for k in range(min(num_samples, len(valid_scg))):
        idx = idx_start + k
        input_scg = torch.tensor(valid_scg[idx], dtype=torch.float32).view(1, seq_len, input_dim).to(device)
        input_pcg = torch.tensor(valid_pcg[idx], dtype=torch.float32).view(1, seq_len, input_dim).to(device)
        target_ecg = valid_ecg[idx]
        with torch.no_grad():
            pred_ecg = model(input_pcg, input_scg).cpu().squeeze().numpy()
        time_axis = np.arange(seq_len) / fs
        corr_sample = np.corrcoef(pred_ecg, target_ecg)[0, 1]
        plot_reconstruction(time_axis, valid_scg[idx], valid_pcg[idx], target_ecg, pred_ecg, corr_sample, k + 1)
        plot_reconstruction_quality(pred_ecg, target_ecg, time_axis, corr_sample,
                                    record_name=record, sample_idx=k + 1, output_dir=plot_dir)

    mean_corr = float(np.mean(correlations)) if correlations else float('nan')
    print(f"\n  Sr. korelacja Pearsona (wszystkie okna): {mean_corr:.4f} +/- {np.std(correlations):.4f}")





def evaluate_full_pipeline(
    model_ecg_path='models/global_best_ecg_model.pth',
    record='CP-01',
    dataset='Zenodo',
    base_data_dir='./data',
    model_name='bilstm_transformer',
    pipeline_name='kaisti'
):
    """
    Pełny pipeline end-to-end:
      1. Wczytanie i preprocessing sygnałów SCG/GCG
      2. Rekonstrukcja EKG przez model głęboki
      3. Detekcja pików R ze zrekonstruowanego EKG
      4. Obliczenie indeksów HRV i porównanie z ground truth

    Pozwala ocenić, czy błędy rekonstrukcji wpływają na metryki kliniczne HRV.
    """
    # Automatyczna detekcja zbioru danych na podstawie nazwy rekordu
    if record.startswith('sub_'):
        dataset = 'IEEE'
    elif record.startswith('b') or record.startswith('m') or record.startswith('p'):
        dataset = 'PhysioNet'
    else:
        dataset = 'Zenodo'

    print("=" * 60)
    print(f"  FULL PIPELINE: {record} ({dataset})")
    print("=" * 60)

    loader = ECGDataLoader(base_data_dir=base_data_dir)
    pre = Preprocessor(fs=TARGET_FS)

    if dataset == 'IEEE':
        df = loader.load_ieee(record=record, format=True)
    elif dataset == 'PhysioNet':
        df = loader.load_physionet(record=record, format=True)
    else:
        df = loader.load_zenodo(record=record, format=True)

    if df is None or df.empty:
        print(f"  Błąd: brak danych dla {record}.")
        return

    print("[3] Przetwarzanie sygnałów...")
    from pipelines import kaisti_pipeline, advanced_filtering_pipeline, aggregate_and_balance_datasets
    from pipelines import ALTERNATIVE_PIPELINES

    PIPELINES = {
        'kaisti':   kaisti_pipeline,
        'advanced': advanced_filtering_pipeline,
        **ALTERNATIVE_PIPELINES
    }
    pipeline_fn = PIPELINES.get(pipeline_name, kaisti_pipeline)
    results = pipeline_fn(df, fs=TARGET_FS)

    fs      = TARGET_FS
    seq_len = SEQ_LEN

    signals = [results['gcg_final'], results['scg_final'], results['ecg_final']]
    windows = pre.extract_windows(
        signals, seq_len=seq_len,
        clean_mask=results['clean_mask'],
        epoch_sec=results['epoch_sec']
    )
    valid_pcg, valid_scg, valid_ecg = windows[0], windows[1], windows[2]

    if len(valid_scg) == 0:
        print("  Brak czystych okien.")
        return

    # Określenie input_dim
    input_dim = 3 if pipeline_name == 'subband' else 1

    # Ładowanie modelu rekonstrukcji
    from models.architectures import ARCHITECTURE_REGISTRY
    ModelClass = ARCHITECTURE_REGISTRY.get(model_name, ECGReconstructionModel)
    model = ModelClass(input_dim=input_dim).to(device)
    
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
        inp_scg = torch.tensor(valid_scg[idx], dtype=torch.float32).view(1, seq_len, input_dim).to(device)
        inp_pcg = torch.tensor(valid_pcg[idx], dtype=torch.float32).view(1, seq_len, input_dim).to(device)
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
        print(f"  {key:<15}: GT={str(gt_v):<10}  Pred={str(pred_v):<10}  diff={diff_str}")

    # Wykresy — każdy model × rekord dostaje własny podfolder
    plot_dir = os.path.join("results", f"{model_name}_{pipeline_name}", record)
    os.makedirs(plot_dir, exist_ok=True)
    t_full = np.arange(len(reconstructed_ecg)) / fs

    _plot_calls = [
        (plot_reconstruction_quality, (reconstructed_ecg, gt_ecg_full, t_full, mean_corr),
         dict(record_name=record, sample_idx=0, output_dir=plot_dir)),
        (plot_poincare, (gt_peaks,),
         dict(fs=fs, label='GT ECG', output_dir=plot_dir, suffix='gt')),
        (plot_poincare, (pred_peaks,),
         dict(fs=fs, label='Reconstructed', output_dir=plot_dir, suffix='pred')),
        (plot_hrv_spectrum, (gt_peaks,),
         dict(fs=fs, label='GT ECG', output_dir=plot_dir, suffix='gt')),
        (plot_hrv_spectrum, (pred_peaks,),
         dict(fs=fs, label='Reconstructed', output_dir=plot_dir, suffix='pred')),
    ]
    for _fn, _args, _kwargs in _plot_calls:
        try:
            _fn(*_args, **_kwargs)
        except Exception as _e:
            print(f"  [WARN plot] {_fn.__name__}: {_e}")

    return {
        'record':    record,
        'dataset':   dataset,
        'mean_corr': mean_corr,
        'hrv_gt':    hrv_gt,
        'hrv_pred':  hrv_pred,
        'n_windows': len(correlations),
    }

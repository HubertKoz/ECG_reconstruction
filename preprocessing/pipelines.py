import numpy as np

from .utils import select_best_axis, select_axis_pca, select_axis_manual, differentiate, normalize
from .filters import butter_bandpass, butter_bandpass_sos, cheby1_bandpass, savitzky_golay_filter
from .artifacts import remove_motion_artifacts
from .detection import envelope_detection, morphological_detection

"""
PRZYKŁAD WYWOŁANIA PIPELINE'U Z KONFIGURACJĄ:
--------------------------------------------
pre = Preprocessor(fs=256)
dfs = [loader.load_zenodo(r) for r in ['CP-01', 'CP-02']]

# Wywołanie z PCA i zaawansowaną filtracją:
balanced_data = pre.aggregate_and_balance(
    dfs,
    seq_len=250,
    advanced=True,
    pipeline_kwargs={
        'select_func': select_axis_pca,
        'epoch_sec': 10
    }
)
"""


def _normalize_masked(signal, clean_mask, n_samples_epoch):
    """Z-score tylko na czystych próbkach; aplikowany do całego sygnału."""
    mask_samples = np.zeros(len(signal), dtype=bool)
    for i, is_clean in enumerate(clean_mask):
        if is_clean:
            mask_samples[i * n_samples_epoch : (i + 1) * n_samples_epoch] = True
    clean_vals = signal[mask_samples]
    if len(clean_vals) == 0 or clean_vals.std() == 0:
        return normalize(signal)
    return (signal - clean_vals.mean()) / (clean_vals.std() + 1e-8)


def kaisti_pipeline(df, fs=256, epoch_sec=10, select_func=select_best_axis):
    """
    Pełny pipeline preprocessingu inspirowany metodą Kaisti (2018).

    Kolejność kroków:
      1. Wybór osi (PCA / best-axis / manual)
      2. Filtracja pasmowoprzepustowa
      3. Usuwanie artefaktów ruchu (FFT power, interpolacja liniowa)
      4. Różniczkowanie (po usunięciu artefaktów — brak wzmacniania skoków)
      5. Detekcja uderzeń
      6. Normalizacja Z-score (statystyki liczone na czystych próbkach)
    """
    # 1. Wybór osi
    scg_raw, scg_info = select_func(df, ['SCG_X', 'SCG_Y', 'SCG_Z', 'SCG'])
    gcg_raw, gcg_info = select_func(df, ['GCG_X', 'GCG_Y', 'GCG_Z', 'GCG'])

    if scg_raw is None:
        return None

    print(f"[Pipeline] SCG: {scg_info} | GCG: {gcg_info}")

    # 2. Wyciąganie EKG
    ecg_raw = None
    if 'ECG_LA_RA' in df.columns:
        ecg_raw = df['ECG_LA_RA'].values
    elif 'ECG' in df.columns:
        ecg_raw = df['ECG'].values

    # 3. Filtracja (SCG/GCG: 0.5–20 Hz; ECG: 0.5–40 Hz)
    scg_f = butter_bandpass(scg_raw, fs)
    gcg_f = butter_bandpass(gcg_raw, fs) if gcg_raw is not None else None
    ecg_f = butter_bandpass(ecg_raw, fs, lowcut=0.5, highcut=40.0) if ecg_raw is not None else None

    # 4. Usuwanie artefaktów PRZED różniczkowaniem (interpolacja, nie zerowanie)
    scg_clean, scg_mask = remove_motion_artifacts(scg_f, fs, epoch_sec=epoch_sec)
    if gcg_f is not None:
        gcg_clean, gcg_mask = remove_motion_artifacts(gcg_f, fs, epoch_sec=epoch_sec)
        clean_mask = scg_mask & gcg_mask
    else:
        gcg_clean = None
        clean_mask = scg_mask

    # 5. Różniczkowanie (sygnał jest już ciągły — brak skokowych artefaktów)
    scg_d = differentiate(scg_clean)
    gcg_d = differentiate(gcg_clean) if gcg_clean is not None else None

    # 6. Detekcja uderzeń
    scg_peaks, _ = envelope_detection(scg_d, fs)
    morph_peaks = morphological_detection(scg_d, fs)

    # 7. Normalizacja Z-score na czystych próbkach
    n_samples_epoch = int(epoch_sec * fs)
    scg_norm = _normalize_masked(scg_d, clean_mask, n_samples_epoch)
    gcg_norm = _normalize_masked(gcg_d, clean_mask, n_samples_epoch) if gcg_d is not None else None
    ecg_norm = normalize(ecg_f) if ecg_f is not None else None

    return {
        'scg_raw': scg_raw,
        'gcg_raw': gcg_raw,
        'scg_f': scg_f,
        'gcg_f': gcg_f,
        'scg_d': scg_d,
        'gcg_d': gcg_d,
        'scg_final': scg_norm,
        'gcg_final': gcg_norm,
        'ecg_final': ecg_norm,
        'peaks_env': scg_peaks,
        'peaks_morph': morph_peaks,
        'clean_mask': clean_mask,
        'epoch_sec': epoch_sec,
        'scg_info': scg_info,
        'gcg_info': gcg_info
    }


def aggregate_and_balance_datasets(dfs, fs=256, pipeline_func=kaisti_pipeline, seq_len=250, **pipeline_kwargs):
    """
    Agreguje wiele rekordów (DataFrames), balansując ich udział w zbiorze okien.
    Zwraca słownik z kluczami 'gcg_final', 'scg_final', 'ecg_final' (numpy arrays).
    """
    from .utils import extract_windows

    all_records_windows = []

    for df in dfs:
        res = pipeline_func(df, fs=fs, **pipeline_kwargs)
        if res is None:
            continue

        signals = [res.get('gcg_final'), res.get('scg_final'), res.get('ecg_final')]
        active_indices = [i for i, s in enumerate(signals) if s is not None]
        active_signals = [signals[i] for i in active_indices]

        if not active_signals:
            continue

        windows = extract_windows(
            active_signals,
            fs,
            seq_len=seq_len,
            clean_mask=res.get('clean_mask'),
            epoch_sec=res.get('epoch_sec', 10)
        )

        if len(windows[0]) > 0:
            full_windows = [None] * 3
            for i, idx in enumerate(active_indices):
                full_windows[idx] = windows[i]
            all_records_windows.append(full_windows)

    if not all_records_windows:
        print("[Aggregate] Błąd: Nie udało się wyekstrahować żadnych okien.")
        return None

    min_n = min([len(w[i]) for w in all_records_windows for i in range(3) if w[i] is not None])
    print(f"[Aggregate] Balansowanie: {len(all_records_windows)} rekordów × {min_n} okien każdy.")

    aggregated = {'gcg_final': [], 'scg_final': [], 'ecg_final': []}
    keys = ['gcg_final', 'scg_final', 'ecg_final']
    for record_wins in all_records_windows:
        for i, key in enumerate(keys):
            if record_wins[i] is not None:
                aggregated[key].append(record_wins[i][:min_n])

    for key in keys:
        if aggregated[key]:
            aggregated[key] = np.concatenate(aggregated[key])
        else:
            aggregated[key] = None

    return aggregated


def advanced_filtering_pipeline(df, fs=256, epoch_sec=10, select_func=select_best_axis):
    """
    Pipeline z filtrami SOS i Czebyszewa. Zachowuje tę samą kolejność kroków co kaisti_pipeline.
    """
    scg_raw, scg_info = select_func(df, ['SCG_X', 'SCG_Y', 'SCG_Z', 'SCG'])
    gcg_raw, gcg_info = select_func(df, ['GCG_X', 'GCG_Y', 'GCG_Z', 'GCG'])

    if scg_raw is None:
        return None

    print(f"[Advanced Pipeline] SCG: {scg_info} | GCG: {gcg_info}")

    ecg_raw = df['ECG_LA_RA'].values if 'ECG_LA_RA' in df.columns else (df['ECG'].values if 'ECG' in df.columns else None)

    # Filtracja SOS / Czebyszew
    scg_f = cheby1_bandpass(scg_raw, fs, lowcut=0.5, highcut=20.0, order=4)
    gcg_f = cheby1_bandpass(gcg_raw, fs, lowcut=0.5, highcut=20.0, order=4) if gcg_raw is not None else None
    ecg_f = butter_bandpass_sos(ecg_raw, fs, lowcut=0.5, highcut=40.0, order=4) if ecg_raw is not None else None

    # Savitzky-Golay na EKG (wygładzenie przed detekcją)
    if ecg_f is not None:
        ecg_f = savitzky_golay_filter(ecg_f, window_length=15, polyorder=3)

    # Usuwanie artefaktów PRZED różniczkowaniem
    scg_clean, scg_mask = remove_motion_artifacts(scg_f, fs, epoch_sec=epoch_sec)
    if gcg_f is not None:
        gcg_clean, gcg_mask = remove_motion_artifacts(gcg_f, fs, epoch_sec=epoch_sec)
        clean_mask = scg_mask & gcg_mask
    else:
        gcg_clean = None
        clean_mask = scg_mask

    scg_d = differentiate(scg_clean)
    gcg_d = differentiate(gcg_clean) if gcg_clean is not None else None

    scg_peaks, _ = envelope_detection(scg_d, fs)
    morph_peaks = morphological_detection(scg_d, fs)

    n_samples_epoch = int(epoch_sec * fs)
    scg_norm = _normalize_masked(scg_d, clean_mask, n_samples_epoch)
    gcg_norm = _normalize_masked(gcg_d, clean_mask, n_samples_epoch) if gcg_d is not None else None
    ecg_norm = normalize(ecg_f) if ecg_f is not None else None

    return {
        'scg_raw': scg_raw,
        'gcg_raw': gcg_raw,
        'scg_f': scg_f,
        'gcg_f': gcg_f,
        'scg_final': scg_norm,
        'gcg_final': gcg_norm,
        'ecg_final': ecg_norm,
        'peaks_env': scg_peaks,
        'peaks_morph': morph_peaks,
        'clean_mask': clean_mask,
        'epoch_sec': epoch_sec,
        'scg_info': scg_info,
        'gcg_info': gcg_info
    }

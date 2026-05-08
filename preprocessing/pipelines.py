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

def kaisti_pipeline(df, fs=256, epoch_sec=10, select_func=select_best_axis):
    """
    Pełny pipeline preprocessingu inspirowany metodą Kaisti.
    """
    # 1. Wybór osi (używamy przekazanej funkcji selekcji)
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
        
    # 3. Filtracja
    scg_f = butter_bandpass(scg_raw, fs)
    gcg_f = butter_bandpass(gcg_raw, fs) if gcg_raw is not None else None
    ecg_f = butter_bandpass(ecg_raw, fs, lowcut=0.5, highcut=40.0) if ecg_raw is not None else None
    
    # 4. Różniczkowanie
    scg_d = differentiate(scg_f)
    gcg_d = differentiate(gcg_f) if gcg_f is not None else None
    
    # 5. Usuwanie artefaktów
    scg_clean, scg_mask = remove_motion_artifacts(scg_d, fs, epoch_sec=epoch_sec)
    if gcg_d is not None:
        gcg_clean, gcg_mask = remove_motion_artifacts(gcg_d, fs, epoch_sec=epoch_sec)
        clean_mask = scg_mask & gcg_mask
    else:
        gcg_clean = None
        clean_mask = scg_mask
        
    # Wspólna maska
    n_samples_epoch = int(epoch_sec * fs)
    for i, is_clean in enumerate(clean_mask):
        if not is_clean:
            scg_clean[i*n_samples_epoch : (i+1)*n_samples_epoch] = 0
            if gcg_clean is not None:
                gcg_clean[i*n_samples_epoch : (i+1)*n_samples_epoch] = 0
                
    # 6. Detekcja uderzeń
    scg_peaks, _ = envelope_detection(scg_clean, fs)
    morph_peaks = morphological_detection(scg_clean, fs)
    
    # 7. Normalizacja
    scg_norm = normalize(scg_clean)
    gcg_norm = normalize(gcg_clean) if gcg_clean is not None else None
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
    Agreguje wiele rekordów (DataFrames), balansując ich udział w ostatecznym zbiorze okien.
    Zwraca ustandaryzowany słownik 'final' z okienami.
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
    print(f"[Aggregate] Balansowanie: Każdy z {len(all_records_windows)} rekordów dostarczy po {min_n} okien.")
    
    aggregated = {
        'gcg_final': [],
        'scg_final': [],
        'ecg_final': []
    }
    
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
    Zaawansowany pipeline wykorzystujący filtry SOS oraz filtr Czebyszewa.
    """
    # 1. Wybór osi
    scg_raw, scg_info = select_func(df, ['SCG_X', 'SCG_Y', 'SCG_Z', 'SCG'])
    gcg_raw, gcg_info = select_func(df, ['GCG_X', 'GCG_Y', 'GCG_Z', 'GCG'])
    
    if scg_raw is None:
        return None
        
    print(f"[Advanced Pipeline] SCG: {scg_info} | GCG: {gcg_info}")
    
    ecg_raw = df['ECG_LA_RA'].values if 'ECG_LA_RA' in df.columns else (df['ECG'].values if 'ECG' in df.columns else None)

    # 2. Filtracja przy użyciu SOS
    scg_f = cheby1_bandpass(scg_raw, fs, lowcut=0.5, highcut=20.0, order=4)
    gcg_f = cheby1_bandpass(gcg_raw, fs, lowcut=0.5, highcut=20.0, order=4) if gcg_raw is not None else None
    ecg_f = butter_bandpass_sos(ecg_raw, fs, lowcut=0.5, highcut=40.0, order=4) if ecg_raw is not None else None

    # 3. Wygładzanie Savitzky-Golay
    if ecg_f is not None:
        ecg_f = savitzky_golay_filter(ecg_f, window_length=15, polyorder=3)

    # 4. Reszta procesu
    scg_d = differentiate(scg_f)
    gcg_d = differentiate(gcg_f) if gcg_f is not None else None
    
    scg_clean, scg_mask = remove_motion_artifacts(scg_d, fs, epoch_sec=epoch_sec)
    if gcg_d is not None:
        gcg_clean, gcg_mask = remove_motion_artifacts(gcg_d, fs, epoch_sec=epoch_sec)
        clean_mask = scg_mask & gcg_mask
    else:
        gcg_clean = None
        clean_mask = scg_mask
        
    n_samples_epoch = int(epoch_sec * fs)
    for i, is_clean in enumerate(clean_mask):
        if not is_clean:
            scg_clean[i*n_samples_epoch : (i+1)*n_samples_epoch] = 0
            if gcg_clean is not None:
                gcg_clean[i*n_samples_epoch : (i+1)*n_samples_epoch] = 0
                
    scg_peaks, _ = envelope_detection(scg_clean, fs)
    morph_peaks = morphological_detection(scg_clean, fs)
    
    scg_norm = normalize(scg_clean)
    gcg_norm = normalize(gcg_clean) if gcg_clean is not None else None
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

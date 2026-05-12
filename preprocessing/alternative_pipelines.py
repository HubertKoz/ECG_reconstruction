"""
Alternatywne pipeline'y preprocessingu sygnałów SCG/GCG → EKG.

Każdy pipeline ma ten sam interfejs co kaisti_pipeline:
  pipeline(df, fs, epoch_sec, select_func) -> dict z kluczami:
    scg_final, gcg_final, ecg_final, clean_mask, epoch_sec, peaks_env, peaks_morph, ...

Dostępne pipeline'y:
  minimal_pipeline   – tylko filtracja BP + normalizacja; brak diff i usuwania artefaktów
  wavelet_pipeline   – wavelet denoising zamiast filtracji BP (pywt lub fallback SG)
  robust_pipeline    – Czebyszew + ściślejsze usuwanie artefaktów + SG; bez diff
"""

import numpy as np

from .utils import select_best_axis, normalize, differentiate
from .filters import butter_bandpass, cheby1_bandpass, savitzky_golay_filter
from .artifacts import remove_motion_artifacts
from .detection import envelope_detection, morphological_detection
from .pipelines import _normalize_masked


# ---------------------------------------------------------------------------
# 1. MINIMAL PIPELINE
# ---------------------------------------------------------------------------

def minimal_pipeline(df, fs=256, epoch_sec=10, select_func=select_best_axis):
    """
    Minimalny preprocessing: tylko filtracja pasmowoprzepustowa i normalizacja.
    Brak różniczkowania i usuwania artefaktów — sprawdza, czy surowy kształt
    sygnału mechanicznego jest wystarczający dla modelu.
    """
    scg_raw, scg_info = select_func(df, ['SCG_X', 'SCG_Y', 'SCG_Z', 'SCG'])
    gcg_raw, gcg_info = select_func(df, ['GCG_X', 'GCG_Y', 'GCG_Z', 'GCG'])

    if scg_raw is None:
        return None

    print(f"[Minimal Pipeline] SCG: {scg_info} | GCG: {gcg_info}")

    ecg_raw = df['ECG_LA_RA'].values if 'ECG_LA_RA' in df.columns else (
              df['ECG'].values if 'ECG' in df.columns else None)

    scg_f = butter_bandpass(scg_raw, fs, lowcut=0.5, highcut=20.0)
    gcg_f = butter_bandpass(gcg_raw, fs, lowcut=0.5, highcut=20.0) if gcg_raw is not None else None
    ecg_f = butter_bandpass(ecg_raw, fs, lowcut=0.5, highcut=40.0) if ecg_raw is not None else None

    # Maska: wszystkie epoki czyste (brak filtracji artefaktów)
    n_epochs = len(scg_f) // int(epoch_sec * fs)
    clean_mask = np.ones(max(n_epochs, 1), dtype=bool)

    # Detekcja uderzeń na surowym sygnale filtrowanym
    scg_peaks, _ = envelope_detection(scg_f, fs)
    morph_peaks  = morphological_detection(scg_f, fs)

    scg_norm = normalize(scg_f)
    gcg_norm = normalize(gcg_f) if gcg_f is not None else None
    ecg_norm = normalize(ecg_f) if ecg_f is not None else None

    return {
        'scg_raw': scg_raw, 'gcg_raw': gcg_raw,
        'scg_f': scg_f, 'gcg_f': gcg_f, 'scg_d': scg_f, 'gcg_d': gcg_f,
        'scg_final': scg_norm, 'gcg_final': gcg_norm, 'ecg_final': ecg_norm,
        'peaks_env': scg_peaks, 'peaks_morph': morph_peaks,
        'clean_mask': clean_mask, 'epoch_sec': epoch_sec,
        'scg_info': scg_info, 'gcg_info': gcg_info
    }


# ---------------------------------------------------------------------------
# 2. WAVELET PIPELINE
# ---------------------------------------------------------------------------

def _wavelet_denoise(signal, wavelet='db4', level=4, mode='soft'):
    """
    Denoise sygnału przez progowanie współczynników falkowych.
    Wymaga pywt; jeśli niedostępne, zwraca sygnał po Savitzky-Golay.
    """
    try:
        import pywt
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        # Próg Donoho–Johnstone (universal threshold)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thr = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs[1:] = [pywt.threshold(c, thr, mode=mode) for c in coeffs[1:]]
        denoised = pywt.waverec(coeffs, wavelet)
        # waverec może zwrócić o 1 próbkę więcej
        return denoised[:len(signal)]
    except ImportError:
        # Fallback: Savitzky-Golay jako wygładzacz
        return savitzky_golay_filter(signal, window_length=21, polyorder=3)


def wavelet_pipeline(df, fs=256, epoch_sec=10, select_func=select_best_axis):
    """
    Pipeline z wavelet denoising zamiast klasycznej filtracji Butterwortha.
    Wavelet lepiej zachowuje morfologię QRS przy silnym szumie.
    Kolejność: wavelet denoise → usuwanie artefaktów → różniczkowanie → normalizacja.
    """
    scg_raw, scg_info = select_func(df, ['SCG_X', 'SCG_Y', 'SCG_Z', 'SCG'])
    gcg_raw, gcg_info = select_func(df, ['GCG_X', 'GCG_Y', 'GCG_Z', 'GCG'])

    if scg_raw is None:
        return None

    print(f"[Wavelet Pipeline] SCG: {scg_info} | GCG: {gcg_info}")

    ecg_raw = df['ECG_LA_RA'].values if 'ECG_LA_RA' in df.columns else (
              df['ECG'].values if 'ECG' in df.columns else None)

    # Wstępna filtracja BP (usuwa DC i szum powyżej Nyquist/2)
    scg_bp = butter_bandpass(scg_raw, fs, lowcut=0.5, highcut=20.0)
    gcg_bp = butter_bandpass(gcg_raw, fs, lowcut=0.5, highcut=20.0) if gcg_raw is not None else None
    ecg_bp = butter_bandpass(ecg_raw, fs, lowcut=0.5, highcut=40.0) if ecg_raw is not None else None

    # Wavelet denoising
    scg_w = _wavelet_denoise(scg_bp)
    gcg_w = _wavelet_denoise(gcg_bp) if gcg_bp is not None else None
    ecg_w = _wavelet_denoise(ecg_bp, wavelet='sym5', level=5) if ecg_bp is not None else None

    # Usuwanie artefaktów (po wavelet — sygnał jest już częściowo wygładzony)
    scg_clean, scg_mask = remove_motion_artifacts(scg_w, fs, epoch_sec=epoch_sec)
    if gcg_w is not None:
        gcg_clean, gcg_mask = remove_motion_artifacts(gcg_w, fs, epoch_sec=epoch_sec)
        clean_mask = scg_mask & gcg_mask
    else:
        gcg_clean = None
        clean_mask = scg_mask

    # Różniczkowanie
    scg_d = differentiate(scg_clean)
    gcg_d = differentiate(gcg_clean) if gcg_clean is not None else None

    scg_peaks, _ = envelope_detection(scg_d, fs)
    morph_peaks  = morphological_detection(scg_d, fs)

    n_samples_epoch = int(epoch_sec * fs)
    scg_norm = _normalize_masked(scg_d, clean_mask, n_samples_epoch)
    gcg_norm = _normalize_masked(gcg_d, clean_mask, n_samples_epoch) if gcg_d is not None else None
    ecg_norm = normalize(ecg_w) if ecg_w is not None else None

    return {
        'scg_raw': scg_raw, 'gcg_raw': gcg_raw,
        'scg_f': scg_bp, 'gcg_f': gcg_bp, 'scg_d': scg_d, 'gcg_d': gcg_d,
        'scg_final': scg_norm, 'gcg_final': gcg_norm, 'ecg_final': ecg_norm,
        'peaks_env': scg_peaks, 'peaks_morph': morph_peaks,
        'clean_mask': clean_mask, 'epoch_sec': epoch_sec,
        'scg_info': scg_info, 'gcg_info': gcg_info
    }


# ---------------------------------------------------------------------------
# 3. ROBUST PIPELINE
# ---------------------------------------------------------------------------

def robust_pipeline(df, fs=256, epoch_sec=10, select_func=select_best_axis):
    """
    Wzmocniony pipeline z filtrem Czebyszewa (ostrzejsze odcięcie)
    i ściślejszym usuwaniem artefaktów (próg 1.1 zamiast 1.25).
    Savitzky-Golay na SCG/GCG zapobiega szumowi po filtracji.
    Bez różniczkowania — model widzi kształt sygnału bezpośrednio.

    Dobrze sprawdza się przy danych z silnymi artefaktami ruchowymi.
    """
    scg_raw, scg_info = select_func(df, ['SCG_X', 'SCG_Y', 'SCG_Z', 'SCG'])
    gcg_raw, gcg_info = select_func(df, ['GCG_X', 'GCG_Y', 'GCG_Z', 'GCG'])

    if scg_raw is None:
        return None

    print(f"[Robust Pipeline] SCG: {scg_info} | GCG: {gcg_info}")

    ecg_raw = df['ECG_LA_RA'].values if 'ECG_LA_RA' in df.columns else (
              df['ECG'].values if 'ECG' in df.columns else None)

    # Czebyszew typ I: ostrzejsze odcięcie niż Butterworth
    scg_f = cheby1_bandpass(scg_raw, fs, lowcut=0.5, highcut=20.0, order=4)
    gcg_f = cheby1_bandpass(gcg_raw, fs, lowcut=0.5, highcut=20.0, order=4) if gcg_raw is not None else None
    ecg_f = cheby1_bandpass(ecg_raw, fs, lowcut=0.5, highcut=40.0, order=4) if ecg_raw is not None else None

    # Wygładzanie SG na sygnałach mechanicznych (zachowuje morfologię pików)
    scg_sg = savitzky_golay_filter(scg_f, window_length=11, polyorder=3)
    gcg_sg = savitzky_golay_filter(gcg_f, window_length=11, polyorder=3) if gcg_f is not None else None
    ecg_sg = savitzky_golay_filter(ecg_f, window_length=15, polyorder=4) if ecg_f is not None else None

    # Ściślejsze usuwanie artefaktów (threshold_p=1.1 vs 1.25 w kaisti)
    scg_clean, scg_mask = remove_motion_artifacts(scg_sg, fs, epoch_sec=epoch_sec, threshold_p=1.1)
    if gcg_sg is not None:
        gcg_clean, gcg_mask = remove_motion_artifacts(gcg_sg, fs, epoch_sec=epoch_sec, threshold_p=1.1)
        clean_mask = scg_mask & gcg_mask
    else:
        gcg_clean = None
        clean_mask = scg_mask

    # Brak różniczkowania — surowy (wygładzony) kształt sygnału
    scg_peaks, _ = envelope_detection(scg_clean, fs)
    morph_peaks  = morphological_detection(scg_clean, fs)

    n_samples_epoch = int(epoch_sec * fs)
    scg_norm = _normalize_masked(scg_clean, clean_mask, n_samples_epoch)
    gcg_norm = _normalize_masked(gcg_clean, clean_mask, n_samples_epoch) if gcg_clean is not None else None
    ecg_norm = normalize(ecg_sg) if ecg_sg is not None else None

    return {
        'scg_raw': scg_raw, 'gcg_raw': gcg_raw,
        'scg_f': scg_f, 'gcg_f': gcg_f, 'scg_d': scg_clean, 'gcg_d': gcg_clean,
        'scg_final': scg_norm, 'gcg_final': gcg_norm, 'ecg_final': ecg_norm,
        'peaks_env': scg_peaks, 'peaks_morph': morph_peaks,
        'clean_mask': clean_mask, 'epoch_sec': epoch_sec,
        'scg_info': scg_info, 'gcg_info': gcg_info
    }


# Słownik wszystkich alternatywnych pipelinów (do importu w compare_all.py)
ALTERNATIVE_PIPELINES = {
    'minimal': minimal_pipeline,
    'wavelet': wavelet_pipeline,
    'robust':  robust_pipeline,
}

from .filters import butter_bandpass, butter_bandpass_sos, savitzky_golay_filter, cheby1_bandpass
from .artifacts import remove_motion_artifacts
from .detection import envelope_detection, morphological_detection, detect_kaisti_peaks
from .utils import select_best_axis, select_axis_pca, select_axis_manual, differentiate, normalize, extract_windows

from config import TARGET_FS, SEQ_LEN


class Preprocessor:
    """Wrapper dla wygody — agreguje najczęściej używane funkcje preprocessingu."""

    def __init__(self, fs: int = TARGET_FS):
        self.fs = fs

    # ── Filtry ────────────────────────────────────────────────────────────────
    def butter_bandpass(self, data, lowcut=0.5, highcut=20.0, order=3):
        return butter_bandpass(data, self.fs, lowcut, highcut, order)

    def butter_bandpass_sos(self, data, lowcut=0.5, highcut=20.0, order=3):
        return butter_bandpass_sos(data, self.fs, lowcut, highcut, order)

    def cheby1_bandpass(self, data, lowcut=0.5, highcut=20.0, order=3, rp=1):
        return cheby1_bandpass(data, self.fs, lowcut, highcut, order, rp)

    def savitzky_golay(self, data, window_length=11, polyorder=3):
        return savitzky_golay_filter(data, window_length, polyorder)

    # ── Artefakty i pomocnicze ────────────────────────────────────────────────
    def differentiate(self, data):
        return differentiate(data)

    def remove_motion_artifacts(self, signal, epoch_sec=10, threshold_p=1.25):
        return remove_motion_artifacts(signal, self.fs, epoch_sec, threshold_p)

    # ── Detekcja uderzeń ──────────────────────────────────────────────────────
    def envelope_detection(self, signal, window_ms=800):
        return envelope_detection(signal, self.fs, window_ms)

    def morphological_detection(self, signal):
        return morphological_detection(signal, self.fs)

    def detect_kaisti_peaks(self, signal, use_fusion=True):
        return detect_kaisti_peaks(signal, self.fs, use_fusion)

    # ── Selekcja osi ──────────────────────────────────────────────────────────
    def select_best_axis(self, df, columns):
        sig, info = select_best_axis(df, columns)
        if sig is not None:
            print(f"[Preprocessor] {info}")
        return sig

    # ── Okna ──────────────────────────────────────────────────────────────────
    def extract_windows(self, signals, seq_len: int = SEQ_LEN, clean_mask=None, epoch_sec=10):
        return extract_windows(signals, self.fs, seq_len, clean_mask, epoch_sec)


__all__ = [
    'Preprocessor',
    'butter_bandpass', 'butter_bandpass_sos', 'savitzky_golay_filter', 'cheby1_bandpass',
    'remove_motion_artifacts',
    'envelope_detection', 'morphological_detection', 'detect_kaisti_peaks',
    'select_best_axis', 'select_axis_pca', 'select_axis_manual',
    'differentiate', 'normalize', 'extract_windows',
]

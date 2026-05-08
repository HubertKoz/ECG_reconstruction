from .filters import butter_bandpass, butter_bandpass_sos, savitzky_golay_filter, cheby1_bandpass
from .artifacts import remove_motion_artifacts
from .detection import envelope_detection, morphological_detection, detect_kaisti_peaks
from .utils import select_best_axis, select_axis_pca, select_axis_manual, differentiate, normalize, extract_windows
from .pipelines import kaisti_pipeline, aggregate_and_balance_datasets, advanced_filtering_pipeline

class Preprocessor:
    """
    Wrapper dla kompatybilności wstecznej z nowymi funkcjonalnościami.
    """
    def __init__(self, fs=256):
        self.fs = fs

    def select_best_axis(self, df, columns):
        # Zachowujemy kompatybilność, ale pod maską używamy nowej logiki
        signal, info = select_best_axis(df, columns)
        if signal is not None:
            print(f"[Preprocessor] {info}")
        return signal

    def butter_bandpass(self, data, lowcut=0.5, highcut=20.0, order=3):
        return butter_bandpass(data, self.fs, lowcut, highcut, order)

    def butter_bandpass_sos(self, data, lowcut=0.5, highcut=20.0, order=3):
        return butter_bandpass_sos(data, self.fs, lowcut, highcut, order)

    def cheby1_bandpass(self, data, lowcut=0.5, highcut=20.0, order=3, rp=1):
        return cheby1_bandpass(data, self.fs, lowcut, highcut, order, rp)

    def savitzky_golay(self, data, window_length=11, polyorder=3):
        return savitzky_golay_filter(data, window_length, polyorder)

    def differentiate(self, data):
        return differentiate(data)

    def remove_motion_artifacts(self, signal, epoch_sec=10, threshold_p=1.25):
        clean_signal, clean_mask = remove_motion_artifacts(signal, self.fs, epoch_sec, threshold_p)
        return clean_signal, clean_mask

    def envelope_detection(self, signal, window_ms=800):
        return envelope_detection(signal, self.fs, window_ms)

    def morphological_detection(self, signal):
        return morphological_detection(signal, self.fs)

    def detect_kaisti_peaks(self, signal, use_fusion=True):
        return detect_kaisti_peaks(signal, self.fs, use_fusion)

    def process_pipeline(self, df, advanced=False, **pipeline_kwargs):
        """
        Uruchamia wybrany pipeline. pipeline_kwargs pozwala na przekazanie np. select_func.
        """
        pipeline = advanced_filtering_pipeline if advanced else kaisti_pipeline
        return pipeline(df, self.fs, **pipeline_kwargs)

    def aggregate_and_balance(self, dfs, seq_len=250, advanced=False, pipeline_kwargs=None):
        """
        Agreguje i balansuje dane. pipeline_kwargs pozwala na konfigurację selekcji osi.
        """
        pipeline = advanced_filtering_pipeline if advanced else kaisti_pipeline
        kwargs = pipeline_kwargs or {}
        return aggregate_and_balance_datasets(dfs, fs=self.fs, pipeline_func=pipeline, seq_len=seq_len, **kwargs)

    def extract_windows(self, signals, seq_len=250, clean_mask=None, epoch_sec=10):
        return extract_windows(signals, self.fs, seq_len, clean_mask, epoch_sec)

__all__ = [
    'Preprocessor',
    'butter_bandpass',
    'butter_bandpass_sos',
    'savitzky_golay_filter',
    'cheby1_bandpass',
    'remove_motion_artifacts',
    'envelope_detection',
    'morphological_detection',
    'detect_kaisti_peaks',
    'select_best_axis',
    'select_axis_pca',
    'select_axis_manual',
    'differentiate',
    'normalize',
    'extract_windows',
    'kaisti_pipeline',
    'aggregate_and_balance_datasets',
    'advanced_filtering_pipeline'
]

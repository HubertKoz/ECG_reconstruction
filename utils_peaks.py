import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def extract_r_peaks(ecg_signal: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """
    Uproszczona implementacja detekcji zespołu QRS z sygnału EKG (podobieństwo do Pan-Tompkins).
    1. Filtracja pasmowoprzepustowa (5-15 Hz wg zaleceń Pan-Tompkins).
    2. Różniczkowanie.
    3. Podniesienie do kwadratu.
    4. Proste ruchome okno całkujące.
    5. Znajdowanie pików za pomocą find_peaks.
    """
    if len(ecg_signal) == 0:
        return np.array([])
        
    # 1. Bandpass filter 5-15 Hz
    filtered = butter_bandpass_filter(ecg_signal, 5.0, 15.0, fs, order=2)
    
    # 2. Różniczkowanie (uwydatnienie zboczy)
    diff = np.diff(filtered)
    diff = np.insert(diff, 0, diff[0]) # wyrównanie długości
    
    # 3. Podniesienie do kwadratu (wszystkie wartości stają się dodatnie, a duże skoki dominują)
    squared = diff ** 2
    
    # 4. Ruchome okno (ok. 150 ms = 0.15 * fs próbek)
    window_sz = int(0.15 * fs)
    if window_sz < 1: window_sz = 1
    window = np.ones(window_sz) / window_sz
    integrated = np.convolve(squared, window, mode='same')
    
    # 5. Znajdowanie szczytów
    # Minimalny odstęp między uderzeniami: np. 300 ms (tętno maks ~200 BPM)
    min_distance = int(0.3 * fs)
    
    # Próg minimalnej amplitudy zintegrowanego sygnału (adaptacyjny)
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    
    peaks, _ = find_peaks(integrated, distance=min_distance, height=threshold)
    
    # Przesunięcie lokalizacji piku z powrotem na największą wartość z oryginalnego filtrowanego EKG
    # (ze względu na bezwładność filtra i całki okna, pik zespolony mógł się przesunąć o parę ms).
    refined_peaks = []
    search_window = int(0.1 * fs) # Wyszukiwanie lokalnego ekstremum w oknach +/- 100 ms
    for p in peaks:
        start = max(0, p - search_window)
        end = min(len(filtered), p + search_window)
        local_max = np.argmax(filtered[start:end])
        refined_peaks.append(start + local_max)
        
    return np.unique(np.array(refined_peaks))

def refine_peak_parabolic(y_vals: np.ndarray, peak_idx: int) -> float:
    """
    Wykorzystuje interpolację paraboliczną trzech punktów wokół szczytu
    do wyznaczenia jego położenia z precyzją sub-próbkową.
    """
    if peak_idx <= 0 or peak_idx >= len(y_vals) - 1:
        return float(peak_idx)
    
    y_m1 = y_vals[peak_idx - 1]
    y_0  = y_vals[peak_idx]
    y_p1 = y_vals[peak_idx + 1]
    
    # Wzór na przesunięcie d od x0:
    # d = 0.5 * (y_m1 - y_p1) / (y_m1 - 2*y_0 + y_p1)
    # Mianownik to druga pochodna; jeśli jest bliska 0, parabola jest płaska
    denom = y_m1 - 2*y_0 + y_p1
    if abs(denom) < 1e-8:
        return float(peak_idx)
    
    d = 0.5 * (y_m1 - y_p1) / denom
    return float(peak_idx + d)

def generate_beat_mask(signal_len: int, peaks: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """
    Przekształca listę indeksów pików na ciągły sygnał prawdopodobieństwa (maskę) za pomocą rozmycia Gaussa.
    """
    mask = np.zeros(signal_len, dtype=np.float32)
    if len(peaks) == 0:
        return mask
        
    valid_peaks = peaks[(peaks >= 0) & (peaks < signal_len)]
    mask[valid_peaks] = 1.0
    
    if sigma > 0:
        # szybkie rozmycie Gaussa 1D
        x = np.arange(-int(4*sigma), int(4*sigma)+1)
        kernel = np.exp(-(x**2)/(2 * sigma**2))
        mask = np.convolve(mask, kernel, mode='same')
        
    # Normalizacja max do 1
    max_val = np.max(mask)
    if max_val > 0:
        mask = mask / max_val
        
    return mask

def get_hr_from_rr(rr_intervals_samples: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """
    Oblicza chwilowe tętno ze zmierzonych odstępów RR (w próbkach).
    BPM = 60 / (RR_próbki / fs)
    """
    rr_sec = rr_intervals_samples / fs
    bpm = 60.0 / rr_sec
    return bpm

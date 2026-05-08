import numpy as np
from scipy.signal import find_peaks, convolve
from scipy.signal.windows import gaussian
from sklearn.cluster import KMeans

def envelope_detection(signal, fs, window_ms=800):
    """
    Detekcja uderzeń metodą obwiedni (Envelope-based).
    """
    squared = signal ** 2
    window_len = int((window_ms / 1000.0) * fs)
    std = window_len / 4
    gauss_win = gaussian(window_len, std)
    gauss_win /= np.sum(gauss_win)
    
    envelope = convolve(squared, gauss_win, mode='same')
    peaks, _ = find_peaks(envelope, distance=int(0.5 * fs))
    return peaks, envelope

def morphological_detection(signal, fs):
    """
    Detekcja uderzeń metodą morfologiczną (Clustering).
    """
    peaks, properties = find_peaks(signal, distance=int(0.4 * fs), height=np.mean(signal))
    
    if len(peaks) < 2:
        return peaks
        
    heights = properties['peak_heights'].reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10).fit(heights)
    labels = kmeans.labels_
    
    if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
        mask = (labels == 0)
    else:
        mask = (labels == 1)
            
    return peaks[mask]

def detect_kaisti_peaks(signal, fs, use_fusion=True):
    """
    Autonomiczna detekcja szczytów inspirowana metodą Kaisti et al. (2018).
    """
    min_dist = int(0.4 * fs)
    candidate_peaks, props = find_peaks(signal, distance=min_dist, height=0.01)
    
    if len(candidate_peaks) < 2:
        return candidate_peaks

    heights = props['peak_heights'].reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=10).fit(np.log1p(heights))
    beat_cluster = np.argmax(kmeans.cluster_centers_)
    morph_peaks = candidate_peaks[kmeans.labels_ == beat_cluster]

    if not use_fusion:
        return morph_peaks

    env_peaks, _ = envelope_detection(signal, fs)

    final_peaks = []
    window = int(0.15 * fs)

    for p_m in morph_peaks:
        if any(np.abs(env_peaks - p_m) <= window):
            final_peaks.append(p_m)

    return np.array(final_peaks)

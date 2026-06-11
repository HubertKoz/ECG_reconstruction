from sklearn.cluster import KMeans
import numpy as np


class HeartbeatClusteringModel:
    """Model K-Means klasyfikujący szczyty SCG na uderzenia serca i szum (metoda Kaisti)."""

    def __init__(self, n_init=10):
        self.model = KMeans(n_clusters=2, n_init=n_init, random_state=42)
        self.is_fitted = False
        self.cluster_centers_ = None

    def fit(self, heights):
        if len(heights) < 2:
            return None
        self.model.fit(heights)
        self.cluster_centers_ = self.model.cluster_centers_
        self.is_fitted = True
        return self

    def predict(self, heights):
        if not self.is_fitted:
            raise ValueError("Model musi być najpierw dopasowany (fit).")
        labels = self.model.predict(heights)
        beat_cluster_idx = np.argmax(self.cluster_centers_)
        return (labels == beat_cluster_idx)


def validate_kaisti_method(detected_peaks, ecg_peaks, fs, window_ms=150):
    """
    Pętla walidacyjna obliczająca metryki z badania (TPR, PPV, RMSE).
    """
    window_samples = int((window_ms / 1000.0) * fs)
    tp = 0
    fp = 0
    fn = 0
    errors_ms = []

    # Kopiujemy listy, aby móc 'odhaczać' dopasowane szczyty
    matched_ecg = [False] * len(ecg_peaks)

    for p in detected_peaks:
        # Szukamy najbliższego szczytu EKG w oknie czasowym
        diffs = np.abs(ecg_peaks - p)
        closest_idx = np.argmin(diffs)
        
        if diffs[closest_idx] <= window_samples:
            tp += 1
            matched_ecg[closest_idx] = True
            # RMSE liczone w milisekundach
            errors_ms.append((diffs[closest_idx] / fs) * 1000)
        else:
            fp += 1

    fn = len(ecg_peaks) - sum(matched_ecg)

    tpr = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    ppv = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
    rmse = np.sqrt(np.mean(np.square(errors_ms))) if errors_ms else 0

    return {"TPR": tpr, "PPV": ppv, "RMSE_ms": rmse}

# --- GŁÓWNA PĘTLA WYKONAWCZA ---

if __name__ == "__main__":
    from dataset import Preprocessor
    from dataset import DataLoader
    from scipy.signal import find_peaks

    loader = DataLoader()
    df_ieee = loader.load_ieee(format=True)

    # 1. Przygotowanie danych
    kp = Preprocessor(fs=256)
    results = kp.process_pipeline(df_ieee) # Dane wejściowe
    
    # 2. Pobieranie cech do ML (Amplitudy szczytów)
    # Wykorzystanie fragmentu funkcji morphological_detection
    raw_signal = results['scg_final']
    candidate_peaks, props = find_peaks(raw_signal, distance=int(0.4 * 256), height=np.mean(raw_signal))
    heights = props['peak_heights'].reshape(-1, 1)

    # 3. 'TRENING' (Dopasowanie K-Means do konkretnego rekordu)
    ml_model = HeartbeatClusteringModel()
    ml_model.fit(heights)
    
    # 4. PREDYKCJA
    is_beat_mask = ml_model.predict(heights)
    final_detected_peaks = candidate_peaks[is_beat_mask]

    # 5. WALIDACJA (Porównanie z EKG)
    # Założenie, że ecg_peaks zostały wyodrębnione z results['ecg_final']
    ecg_peaks, _ = find_peaks(results['ecg_final'], distance=int(0.5 * 256), height=1.0)
    
    metrics = validate_kaisti_method(final_detected_peaks, ecg_peaks, fs=256)

    print(f"--- WYNIKI ML (K-MEANS) ---")
    print(f"Czułość (TPR): {metrics['TPR']:.2f}%")
    print(f"Precyzja (PPV): {metrics['PPV']:.2f}%")
    print(f"Błąd czasowy (RMSE): {metrics['RMSE_ms']:.2f} ms")
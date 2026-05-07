import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, convolve, find_peaks
from scipy.signal.windows import gaussian
from scipy.fft import fft
from sklearn.cluster import KMeans

class Preprocessor:
    """
    Implementacja zaawansowanego preprocessingu i detekcji uderzeń serca (heartbeat detection) 
    na podstawie pracy: Kaisti, M., et al. (2018). "Stand-alone heartbeat detection in 
    multidimensional mechanocardiograms".

    Klasa ta jest zoptymalizowana pod zbiór IEEE DataPort, wykorzystując zarówno sygnały 
    akcelerometryczne (SCG), jak i żyroskopowe (GCG). Celem jest uzyskanie czystego sygnału 
    mechanicznego, który pozwoli na precyzyjną rekonstrukcję interwałów RR.
    """
    
    def __init__(self, fs=256):
        self.fs = fs


    def select_best_axis(self, df, columns):
        """
        Wybiera oś o najlepszym stosunku amplitudy międzyszczytowej do szumu spośród dostępnych kanałów.
        """
        best_ratio = -1
        best_col = None
        
        for col in columns:
            if col not in df.columns:
                continue
            
            data = df[col].values
            # P2P - miara siły sygnału (Peak-to-Peak)
            p2p = np.ptp(data)
            # MAD - miara poziomu szumu (Median Absolute Deviation)
            mad = np.median(np.abs(data - np.median(data)))
            
            if mad == 0:
                continue
                
            # Stosunek Sygnał/Szum (SNR) wg Kaisti
            ratio = p2p / mad
            if ratio > best_ratio:
                best_ratio = ratio
                best_col = col
        
        print(f"[Kaisti] Wybrano najlepszą oś: {best_col} (Ratio: {best_ratio:.2f})")
        return df[best_col].values if best_col else None



    def butter_bandpass(self, data, lowcut=0.5, highcut=20.0, order=3):
        """
        Filtr Butterwortha IIR 3. rzędu.

        Przejśc na przestrzeń stanów albo system bloków 2. rzędowych sos

        miara: noise to signal ratio
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)



    def differentiate(self, data):
        """
        Obliczenie pochodnej.
        """
        return np.diff(data, prepend=data[0])



    def remove_motion_artifacts(self, signal, epoch_sec=10, threshold_p=1.25):
        """
        Usuwa fragmenty sygnału zanieczyszczone ruchem (sygnał o bardzo wysokiej amplitudzie).
        """
        n_samples_epoch = int(epoch_sec * self.fs)
        n_epochs = len(signal) // n_samples_epoch
        
        if n_epochs == 0:
            return signal, np.array([0])
            
        epoch_powers = []
        for i in range(n_epochs):
            epoch = signal[i*n_samples_epoch : (i+1)*n_samples_epoch]
            # Obliczenie mocy widmowej za pomocą FFT
            freq_data = np.abs(fft(epoch))
            total_power = np.sum(freq_data**2)
            epoch_powers.append(total_power)
            
        median_power = np.median(epoch_powers)
        # Maska czystych danych (poniżej 125% mediany)
        clean_mask = np.array(epoch_powers) <= (threshold_p * median_power)
        
        clean_signal = signal.copy()
        for i, is_clean in enumerate(clean_mask):
            if not is_clean:
                # Wyzerowanie odcinków z szumem
                clean_signal[i*n_samples_epoch : (i+1)*n_samples_epoch] = 0
                
        print(f"[Kaisti] Artifact Removal: Pozostawiono {np.sum(clean_mask)} z {n_epochs} epok.")
        return clean_signal, clean_mask



    def envelope_detection(self, signal, window_ms=800):
        """
        Detekcja uderzeń metodą obwiedni (Envelope-based).
        
        Dlaczego to stosujemy?
        Kwadratowanie sygnału i splot z oknem Gaussa wygładza surowy, oscylacyjny sygnał MCG 
        do postaci "pagórków" (obwiedni). Dzięki temu zamiast szukać mikroszczytków, 
        lokalizujemy środek każdego uderzenia serca.
        """
        # Potęgowanie wzmacnia szczyty względem tła
        squared = signal ** 2
        
        # Okno Gaussa (800ms) - typowy czas trwania cyklu pracy serca
        window_len = int((window_ms / 1000.0) * self.fs)
        std = window_len / 4
        gauss_win = gaussian(window_len, std)
        gauss_win /= np.sum(gauss_win)
        
        # Wygładzanie splotem
        envelope = convolve(squared, gauss_win, mode='same')
        # Detekcja maksimów w obwiedni
        peaks, _ = find_peaks(envelope, distance=int(0.5 * self.fs))
        return peaks, envelope



    def morphological_detection(self, signal):
        """
        Detekcja uderzeń metodą morfologiczną (Clustering).
        
        Dlaczego to stosujemy?
        Amplituda uderzeń serca jest relatywnie stała w krótkim czasie. Grupowanie K-Means 
        pozwala automatycznie rozdzielić wykryte lokalne maxima na dwie grupy: 
        "prawdziwe uderzenia" (wysoka amplituda) oraz "szum/artefakty" (niska amplituda).
        """
        # Wstępne wyłapanie wszystkich potencjalnych szczytów
        peaks, properties = find_peaks(signal, distance=int(0.4 * self.fs), height=np.mean(signal))
        heights = properties['peak_heights'].reshape(-1, 1)
        
        if len(heights) < 2:
            return peaks
            
        # Klasteryzacja na 2 grupy (uderzenia vs szum)
        kmeans = KMeans(n_clusters=2, n_init=10).fit(heights)
        labels = kmeans.labels_
        
        # Wybieramy grupę o wyższej średniej wysokości (prawdziwe bicie serca)
        if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
            mask = (labels == 0)
        else:
            mask = (labels == 1)
            
        return peaks[mask]



    def detect_kaisti_peaks(self, signal, use_fusion=True):
        """
        Autonomiczna detekcja szczytów inspirowana metodą Kaisti et al. (2018).
        Można jej używać zamiennie z scipy.signal.find_peaks.
        
        Zwraca: 
            peaks (np.ndarray): Indeksy wykrytych uderzeń serca.
        """
        # 1. ŚCIEŻKA ML (Morfologiczna / K-Means)
        # Pobieramy kandydatów z bardzo niskim progiem, by ML miał z czego wybierać
        min_dist = int(0.4 * self.fs)
        candidate_peaks, props = find_peaks(signal, distance=min_dist, height=0.01)
        
        if len(candidate_peaks) < 2:
            return candidate_peaks

        # Klastrowanie amplitud (ML)
        heights = props['peak_heights'].reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, n_init=10).fit(np.log1p(heights))
        beat_cluster = np.argmax(kmeans.cluster_centers_)
        morph_peaks = candidate_peaks[kmeans.labels_ == beat_cluster]

        if not use_fusion:
            return morph_peaks

        # 2. ŚCIEŻKA MATEMATYCZNA (Obwiednia / Envelope)
        # Wykorzystujemy Twoją istniejącą funkcję lub jej logikę
        env_peaks, _ = self.envelope_detection(signal)

        # 3. FUZJA (Matching)
        # Kaisti sugeruje, że uderzenie jest pewne, gdy obie metody je widzą
        # w oknie +/- 150ms.
        final_peaks = []
        window = int(0.15 * self.fs)

        for p_m in morph_peaks:
            # Sprawdź czy w pobliżu p_m jest szczyt z metody obwiedni
            if any(np.abs(env_peaks - p_m) <= window):
                final_peaks.append(p_m)

        return np.array(final_peaks)




    def process_pipeline(self, df):
        """
        Główny pipeline preprocessingu.
        """
        # Automatyczny wybór najczystszych osi
        scg_raw = self.select_best_axis(df, ['SCG_X', 'SCG_Y', 'SCG_Z', 'SCG'])
        gcg_raw = self.select_best_axis(df, ['GCG_X', 'GCG_Y', 'GCG_Z', 'GCG'])
        
        if scg_raw is None:
            return None
            
        # Wyciąganie EKG do wspólnej normalizacji i filtracji
        ecg_raw = None
        if 'ECG_LA_RA' in df.columns:
            ecg_raw = df['ECG_LA_RA'].values
        elif 'ECG' in df.columns:
            ecg_raw = df['ECG'].values
            
        # Usuwanie niepożądanych częstotliwości z sygnałów mechanicznych (0.5 - 20 Hz)
        scg_f = self.butter_bandpass(scg_raw)
        gcg_f = self.butter_bandpass(gcg_raw) if gcg_raw is not None else None
        
        # Filtracja EKG (0.5 - 40 Hz) - redukcja dryftu bazowego
        ecg_f = None
        if ecg_raw is not None:
            ecg_f = self.butter_bandpass(ecg_raw, lowcut=0.5, highcut=40.0)
        
        # Uwydatnienie punktów uderzeń serca
        scg_d = self.differentiate(scg_f)
        gcg_d = self.differentiate(gcg_f) if gcg_f is not None else None
        
        # 4. Eliminacja fragmentów z artefaktami ruchu
        epoch_sec = 10
        scg_clean, scg_mask = self.remove_motion_artifacts(scg_d, epoch_sec=epoch_sec)
        
        if gcg_d is not None:
            gcg_clean, gcg_mask = self.remove_motion_artifacts(gcg_d, epoch_sec=epoch_sec)
            # Fuzja masek - okno jest czyste tylko jeśli obie osie są czyste
            clean_mask = scg_mask & gcg_mask
        else:
            gcg_clean = None
            clean_mask = scg_mask
            
        # Zastosuj wspólną maskę, by zerować w obu kanałach dokładnie te same epoki
        n_epochs = len(clean_mask)
        n_samples_epoch = int(epoch_sec * self.fs)
        for i, is_clean in enumerate(clean_mask):
            if not is_clean:
                scg_clean[i*n_samples_epoch : (i+1)*n_samples_epoch] = 0
                if gcg_clean is not None:
                    gcg_clean[i*n_samples_epoch : (i+1)*n_samples_epoch] = 0

        # 5. Detekcja uderzeń przy użyciu fuzji dwóch metod (Ensemble)
        scg_peaks, _ = self.envelope_detection(scg_clean)
        morph_peaks = self.morphological_detection(scg_clean)
        
        # 6. Standaryzacja (Z-score) wyników do uczenia modelu
        def normalize(x): return (x - np.mean(x)) / (np.std(x) + 1e-8)
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
            'scg_kaisti': scg_norm,
            'gcg_kaisti': gcg_norm,
            'ecg_kaisti': ecg_norm, 
            'peaks_env': scg_peaks,
            'peaks_morph': morph_peaks,
            'clean_mask': clean_mask,
            'epoch_sec': epoch_sec
        }













from data_loader import DataLoader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 1. Inicjalizacja loadera i załadowanie prawdziwych danych IEEE
    print("--- Test Kaisti Preprocessor na danych Zenodo ---")
    loader = DataLoader(base_data_dir="./data")

    # Ładujemy pierwszy rekord (np. sub_1)
    df_ieee = loader.load_zenodo()
    #df_ieee = loader.load_ieee()
    if df_ieee is not None:
        # 2. Uruchomienie preprocessingu Kaisti     
        # Dane IEEE mają fs=800Hz wg dokumentacji/read_ieee
        fs = 256
        kp = Preprocessor(fs=fs)
        results = kp.process_pipeline(df_ieee)
        
        if results:
            print(f"Liczba wykrytych uderzeń (Envelope): {len(results['peaks_env'])}")
            print(f"Liczba wykrytych uderzeń (Morphological): {len(results['peaks_morph'])}")
            
            # 3. Wizualizacja wyników
            # alltime
            #skip_sec, duration_sec = 0,450
            # window
            #skip_sec, duration_sec = 20, 10
            skip_sec, duration_sec = 200, 10
            # Parametry okna do wyświetlenia (np. od 5 do 10 sekundy)
            # plot_ieee_signals usunięto w ramach reorganizacji do ocen modularnych
            print("Zakończono preprocessing!")
        else:
            print("Błąd podczas przetwarzania potoku Kaisti.")
    else:
        print(f"Nie udało się załadować rekordu")




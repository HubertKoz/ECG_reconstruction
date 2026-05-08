import numpy as np
import matplotlib.pyplot as plt

def calculate_hrv_indices(r_peaks, fs=256):
    """
    Oblicza podstawowe indeksy HRV na podstawie lokalizacji pików.
    """
    if len(r_peaks) < 2:
        return {'HeartRate': 0, 'SDNN': 0, 'RMSSD': 0}
        
    rr_intervals_samples = np.diff(r_peaks)
    rr_intervals_ms = (rr_intervals_samples / fs) * 1000.0
    
    # Usuwanie anomalii fizjologicznych (np. błędy rzędu 2 sekund między pobudzeniami)
    valid_rr = rr_intervals_ms[(rr_intervals_ms > 300) & (rr_intervals_ms < 2000)]
    
    if len(valid_rr) < 2:
         return {'HeartRate': 0, 'SDNN': 0, 'RMSSD': 0}
         
    mean_rr = np.mean(valid_rr)
    hr = 60000.0 / mean_rr
    
    sdnn = np.std(valid_rr, ddof=1)
    
    diff_rr = np.diff(valid_rr)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    
    return {
        'HeartRate': np.round(hr, 1),
        'SDNN': np.round(sdnn, 2),
        'RMSSD': np.round(rmssd, 2)
    }

def plot_reconstruction(time_axis, valid_scg, valid_pcg, target_ecg, pred_ecg, corr, sample_idx):
    """
    Rysuje wykresy dla rekonstrukcji EKG.
    """
    plt.figure(figsize=(15, 8))
    
    # Subplot 1: Wejścia (SCG i GCG)
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, valid_scg, label="Wejście: SCG (Z-score)", color='#1f77b4', linewidth=1.5)
    plt.plot(time_axis, valid_pcg, label="Wejście: GCG (Z-score)", color='#ff7f0e', linewidth=1.5, alpha=0.8)
    plt.title(f"Sygnały Mechaniczne (Wejście do Modelu) - Wynik Korelacji z EKG: {corr:.4f}")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda znormalizowana")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Subplot 2: Przewidziane vs Prawdziwe EKG
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, target_ecg, label="Referencyjne EKG (Ground Truth)", color='black', linewidth=2.0)
    plt.plot(time_axis, pred_ecg, label="Zrekonstruowane EKG (Predykcja Modelu)", color='red', linestyle='--', linewidth=2.0)
    plt.title(f"Rekonstrukcja fali kardiologicznej (Wyjście z Modelu) - Próbka #{sample_idx}")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda znormalizowana")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_hrv_comparison(t, ecg_full, disp_gt_peaks, scg_plot_norm, disp_pred_peaks, record_name, fs=256):
    """
    Rysuje wykresy porównawcze dla detekcji pików i HRV.
    """
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, ecg_full, color='red', label='Referencyjne EKG')
    plt.plot(disp_gt_peaks / float(fs), ecg_full[(disp_gt_peaks - (t[0]*fs)).astype(int)], 'rx', markersize=10, label='GT R-Peaks')
    plt.title('Referencja (Ground Truth)')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(t, scg_plot_norm, color='blue', alpha=0.6, label='Sygnał SCG (Z-normalized)')
    local_peaks_idx = (disp_pred_peaks - (t[0]*fs)).astype(int)
    plt.plot(disp_pred_peaks / float(fs), scg_plot_norm[local_peaks_idx], 'gx', markersize=10, label='Predykcje Sieci (HR)')
    plt.title('Detekcja mechaniczna modelu (Sygnał znormalizowany)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, ecg_full, color='gray', alpha=0.3, label='Referencyjne EKG')
    plt.vlines(disp_gt_peaks / float(fs), -2, 2, colors='r', linestyles='--', label='GT Peaks (ECG)')
    plt.vlines(disp_pred_peaks / float(fs), -2, 2, colors='g', linestyles='-', label='Pred Peaks (Model)')
    plt.title('Synchronizacja czasowa pików: Referencja vs Predykcja')
    plt.xlabel('Czas (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"hrv_eval_{record_name}.png")
    plt.show()
    print(f"\nWygenerowano i zapisano plik z wykresem 'hrv_eval_{record_name}.png'.")

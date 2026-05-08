import numpy as np
from scipy.fft import fft

def remove_motion_artifacts(signal, fs, epoch_sec=10, threshold_p=1.25):
    """
    Usuwa fragmenty sygnału zanieczyszczone ruchem (sygnał o bardzo wysokiej amplitudzie).
    """
    n_samples_epoch = int(epoch_sec * fs)
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
            
    return clean_signal, clean_mask

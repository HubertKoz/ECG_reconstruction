import numpy as np
from scipy.fft import fft


def remove_motion_artifacts(signal, fs, epoch_sec=10, threshold_p=1.25):
    """
    Wykrywa i interpoluje epoki zanieczyszczone ruchem na podstawie mocy widmowej.

    Moc liczona jako znormalizowana energia FFT (Parseval): sum(|X|^2) / N,
    co zapewnia niezależność progu od długości okna.

    Zanieczyszczone epoki są zastępowane interpolacją liniową między
    ostatnią czystą próbką przed epoką a pierwszą czystą po niej,
    aby uniknąć nieciągłości skokowych po wyzerowaniu.

    Zwraca:
        clean_signal: sygnał z interpolowanymi artefaktami  (float64 ndarray)
        clean_mask:   bool ndarray, True = czysta epoka
    """
    n_samples_epoch = int(epoch_sec * fs)
    n_epochs = len(signal) // n_samples_epoch

    if n_epochs == 0:
        return signal.copy(), np.array([True])

    epoch_powers = []
    for i in range(n_epochs):
        epoch = signal[i * n_samples_epoch : (i + 1) * n_samples_epoch]
        freq_data = np.abs(fft(epoch))
        # Moc znormalizowana przez N — niezależna od długości okna
        total_power = np.sum(freq_data ** 2) / len(epoch)
        epoch_powers.append(total_power)

    median_power = np.median(epoch_powers)
    clean_mask = np.array(epoch_powers) <= (threshold_p * median_power)

    clean_signal = signal.copy().astype(float)

    for i, is_clean in enumerate(clean_mask):
        if is_clean:
            continue

        seg_start = i * n_samples_epoch
        seg_end = (i + 1) * n_samples_epoch

        # Szukamy wartości brzegowych do interpolacji
        # Lewa granica: ostatnia próbka czystej epoki poprzedniej (lub 0.0)
        left_val = clean_signal[seg_start - 1] if seg_start > 0 else 0.0
        # Prawa granica: pierwsza próbka czystej epoki następnej (lub 0.0)
        right_val = clean_signal[seg_end] if seg_end < len(clean_signal) else 0.0

        # Interpolacja liniowa wewnątrz brudnej epoki
        clean_signal[seg_start:seg_end] = np.linspace(left_val, right_val, seg_end - seg_start)

    return clean_signal, clean_mask

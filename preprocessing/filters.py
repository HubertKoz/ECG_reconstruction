from scipy.signal import butter, filtfilt, savgol_filter, cheby1, sosfiltfilt

def butter_bandpass(data, fs, lowcut=0.5, highcut=20.0, order=3):
    """
    Filtr Butterwortha IIR 3. rzędu (klasyczny b, a).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def butter_bandpass_sos(data, fs, lowcut=0.5, highcut=20.0, order=3):
    """
    Filtr Butterwortha wykorzystujący sekcje drugiego rzędu (SOS) dla lepszej stabilności numerycznej.
    """
    sos = butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
    return sosfiltfilt(sos, data)

def savitzky_golay_filter(data, window_length=11, polyorder=3):
    """
    Filtr Savitzky-Golay (często kojarzony z rosyjską szkołą przetwarzania sygnałów dla wygładzania).
    Dobry do zachowania kształtu szczytów.
    """
    return savgol_filter(data, window_length, polyorder)

def cheby1_bandpass(data, fs, lowcut=0.5, highcut=20.0, order=3, rp=1):
    """
    Filtr Czebyszewa typu I (Chebyshev Type I) z tętnieniami w paśmie przepustowym.
    Ma ostrzejsze odcięcie niż Butterworth.
    """
    sos = cheby1(order, rp, [lowcut, highcut], fs=fs, btype='band', output='sos')
    return sosfiltfilt(sos, data)

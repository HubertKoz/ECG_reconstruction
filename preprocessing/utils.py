import numpy as np
from sklearn.decomposition import PCA

def select_best_axis(df, columns):
    """
    Wybiera oś o najlepszym stosunku amplitudy międzyszczytowej do szumu.
    Zwraca: (signal, info_string)
    """
    best_ratio = -1
    best_col = None
    
    for col in columns:
        if col not in df.columns:
            continue
        
        data = df[col].values
        p2p = np.ptp(data)
        mad = np.median(np.abs(data - np.median(data)))
        
        if mad == 0:
            continue
            
        ratio = p2p / mad
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = col
    
    if best_col:
        return df[best_col].values, f"{best_col} (Ratio: {best_ratio:.2f})"
    return None, "None"

def select_axis_pca(df, columns):
    """
    Wykonuje analizę składowych głównych (PCA) i zwraca pierwszą składową.
    """
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        return None, "None"
    
    data = df[valid_cols].values
    # Standaryzacja przed PCA jest zalecana
    data_std = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    
    pca = PCA(n_components=1)
    projected = pca.fit_transform(data_std).flatten()
    
    # Próba zachowania znaku (by SCG/GCG nie było odwrócone losowo)
    # Sprawdzamy korelację z pierwotną osią Z (często dominującą)
    if 'SCG_Z' in valid_cols:
        orig_z = data_std[:, valid_cols.index('SCG_Z')]
        if np.corrcoef(projected, orig_z)[0, 1] < 0:
            projected = -projected
            
    return projected, f"PCA ({', '.join(valid_cols)})"

def select_axis_manual(df, columns, target_axis='SCG_Z'):
    """
    Zwraca wybraną ręcznie oś.
    """
    if target_axis in df.columns:
        return df[target_axis].values, target_axis
    return None, "None"

def differentiate(data):
    """
    Obliczenie pierwszej pochodnej (różnica skończona).
    Pierwsza próbka uzupełniana jest forward-difference (data[1] - data[0]),
    aby zachować długość sygnału bez wprowadzania stałej wartości brzegowej.
    """
    first_diff = data[1] - data[0] if len(data) > 1 else 0.0
    return np.diff(data, prepend=data[0] - first_diff)

def normalize(x):
    """
    Standaryzacja Z-score.
    """
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

def extract_windows(signals, fs, seq_len=250, clean_mask=None, epoch_sec=10):
    """
    Tnie sygnały na okna o stałej długości, opcjonalnie odrzucając te zanieczyszczone.
    """
    n_samples_total = len(signals[0])
    n_windows = n_samples_total // seq_len
    n_samples_epoch = int(epoch_sec * fs)
    
    windows = [[] for _ in range(len(signals))]
    
    for i in range(n_windows):
        start = i * seq_len
        end = start + seq_len
        
        if clean_mask is not None:
            epoch_start = start // n_samples_epoch
            epoch_end = (end - 1) // n_samples_epoch
            if epoch_start >= len(clean_mask) or epoch_end >= len(clean_mask) or not clean_mask[epoch_start] or not clean_mask[epoch_end]:
                continue
                
        for j, sig in enumerate(signals):
            windows[j].append(sig[start:end])
            
    return [np.array(w) for w in windows]

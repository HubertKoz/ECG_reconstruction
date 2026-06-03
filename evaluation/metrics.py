import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Wyłącza blokujące okna GUI w konsoli CLI
import matplotlib.pyplot as plt
from scipy.signal import welch

try:
    from scipy.integrate import trapezoid
except ImportError:
    trapezoid = np.trapz


# ---------------------------------------------------------------------------
# HRV INDICES
# ---------------------------------------------------------------------------

def calculate_hrv_indices(r_peaks, fs=256):
    """
    Oblicza standardowe indeksy HRV: time-domain, frequency-domain i nieliniowe.

    Parametry
    ---------
    r_peaks : array-like
        Pozycje pików R w próbkach.
    fs : float
        Częstotliwość próbkowania [Hz].

    Zwraca
    ------
    dict z kluczami (lub np.nan gdy za mało danych):
      Time-domain : MeanRR, SDNN, RMSSD, pNN50, NN50, CVNN, HeartRate
      Frequency   : VLF_ms2, LF_ms2, HF_ms2, LF_HF_ratio, Total_ms2
      Nonlinear   : SD1, SD2, SD1_SD2_ratio
    """
    nan_result = {
        'HeartRate': np.nan, 'MeanRR': np.nan,
        'SDNN': np.nan, 'RMSSD': np.nan,
        'pNN50': np.nan, 'NN50': np.nan, 'CVNN': np.nan,
        'VLF_ms2': np.nan, 'LF_ms2': np.nan, 'HF_ms2': np.nan,
        'LF_HF_ratio': np.nan, 'Total_ms2': np.nan,
        'SD1': np.nan, 'SD2': np.nan, 'SD1_SD2_ratio': np.nan
    }

    r_peaks = np.asarray(r_peaks)
    if len(r_peaks) < 2:
        return nan_result

    rr_ms = np.diff(r_peaks) / fs * 1000.0

    # Odrzucenie interwałów poza zakresem fizjologicznym (30–200 BPM)
    valid_mask = (rr_ms > 300) & (rr_ms < 2000)
    n_rejected = int(np.sum(~valid_mask))
    if n_rejected > 0:
        print(f"  [HRV] Odrzucono {n_rejected}/{len(rr_ms)} interwałów RR poza zakresem 300–2000 ms.")
    rr = rr_ms[valid_mask]

    if len(rr) < 2:
        return nan_result

    # --- Time-domain ---
    mean_rr = np.mean(rr)
    hr      = 60000.0 / mean_rr
    sdnn    = np.std(rr, ddof=1)
    diff_rr = np.diff(rr)
    rmssd   = np.sqrt(np.mean(diff_rr ** 2))
    nn50    = int(np.sum(np.abs(diff_rr) > 50.0))
    pnn50   = (nn50 / len(diff_rr)) * 100.0 if len(diff_rr) > 0 else np.nan
    cvnn    = (sdnn / mean_rr) * 100.0 if mean_rr > 0 else np.nan

    # --- Frequency-domain (Welch PSD na interpolowanym sygnale RR) ---
    vlf_ms2 = lf_ms2 = hf_ms2 = total_ms2 = lf_hf = np.nan
    if len(rr) >= 8:
        # Interpolacja równomiernie próbkowanego sygnału RR (fs_rr = 4 Hz)
        fs_rr = 4.0
        t_rr = np.cumsum(rr) / 1000.0  # czasy bić w sekundach
        t_rr -= t_rr[0]
        t_interp = np.arange(0, t_rr[-1], 1.0 / fs_rr)
        rr_interp = np.interp(t_interp, t_rr, rr)

        nperseg = min(256, len(rr_interp))
        freqs, psd = welch(rr_interp, fs=fs_rr, nperseg=nperseg)
        df = freqs[1] - freqs[0]

        # Pasma HRV (ms²)
        vlf_mask = (freqs >= 0.0033) & (freqs < 0.04)
        lf_mask  = (freqs >= 0.04)   & (freqs < 0.15)
        hf_mask  = (freqs >= 0.15)   & (freqs < 0.40)

        vlf_ms2   = float(trapezoid(psd[vlf_mask], freqs[vlf_mask])) if vlf_mask.any() else 0.0
        lf_ms2    = float(trapezoid(psd[lf_mask],  freqs[lf_mask]))  if lf_mask.any()  else 0.0
        hf_ms2    = float(trapezoid(psd[hf_mask],  freqs[hf_mask]))  if hf_mask.any()  else 0.0
        total_ms2 = vlf_ms2 + lf_ms2 + hf_ms2
        lf_hf     = (lf_ms2 / hf_ms2) if hf_ms2 > 0 else np.nan

    # --- Nonlinear (Poincaré SD1/SD2) ---
    sd1 = sd2 = sd1_sd2 = np.nan
    if len(rr) >= 3:
        diff_rr2 = np.diff(rr)
        sd1 = np.sqrt(0.5) * np.std(diff_rr2, ddof=1)
        sd2 = np.sqrt(2.0 * np.std(rr, ddof=1) ** 2 - 0.5 * np.std(diff_rr2, ddof=1) ** 2)
        sd1_sd2 = (sd1 / sd2) if sd2 > 0 else np.nan

    return {
        'HeartRate':    round(float(hr),    1),
        'MeanRR':       round(float(mean_rr), 1),
        'SDNN':         round(float(sdnn),  2),
        'RMSSD':        round(float(rmssd), 2),
        'pNN50':        round(float(pnn50), 1),
        'NN50':         int(nn50),
        'CVNN':         round(float(cvnn),  2),
        'VLF_ms2':      round(float(vlf_ms2),   2),
        'LF_ms2':       round(float(lf_ms2),    2),
        'HF_ms2':       round(float(hf_ms2),    2),
        'LF_HF_ratio':  round(float(lf_hf),     3) if not np.isnan(lf_hf) else np.nan,
        'Total_ms2':    round(float(total_ms2),  2),
        'SD1':          round(float(sd1), 2),
        'SD2':          round(float(sd2), 2),
        'SD1_SD2_ratio': round(float(sd1_sd2), 3) if not np.isnan(sd1_sd2) else np.nan,
    }


# ---------------------------------------------------------------------------
# WYKRESY
# ---------------------------------------------------------------------------

def plot_reconstruction(time_axis, valid_scg, valid_pcg, target_ecg, pred_ecg, corr, sample_idx):
    """Sygnały wejściowe + rekonstrukcja EKG vs ground truth."""
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time_axis, valid_scg, label="Wejście: SCG (Z-score)", color='#1f77b4', linewidth=1.5)
    plt.plot(time_axis, valid_pcg, label="Wejście: GCG (Z-score)", color='#ff7f0e', linewidth=1.5, alpha=0.8)
    plt.title(f"Sygnały mechaniczne (wejście) — korelacja z EKG: {corr:.4f}")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda znormalizowana")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_axis, target_ecg, label="Referencyjne EKG (Ground Truth)", color='black', linewidth=2.0)
    plt.plot(time_axis, pred_ecg,   label="Zrekonstruowane EKG (model)",    color='red', linestyle='--', linewidth=2.0)
    plt.title(f"Rekonstrukcja fali EKG — próbka #{sample_idx}  (Pearson r = {corr:.4f})")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda znormalizowana")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_reconstruction_quality(pred_ecg, gt_ecg, t, corr, record_name='', sample_idx=0, save=True):
    """
    Overlay zrekonstruowanego EKG z ground truth + błąd bezwzględny.
    Korelacja Pearsona wyświetlana w tytule.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(t, gt_ecg,   color='black', linewidth=1.8, label='Ground Truth EKG')
    axes[0].plot(t, pred_ecg, color='red',   linewidth=1.5, linestyle='--', label='Rekonstrukcja (model)')
    axes[0].set_title(f"Jakość rekonstrukcji EKG  {record_name}  #{sample_idx}  —  Pearson r = {corr:.4f}")
    axes[0].set_ylabel("Amplituda (z-score)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(t, np.abs(pred_ecg - gt_ecg), color='orange', alpha=0.6, label='|błąd|')
    axes[1].set_ylabel("|błąd bezwzgl.|")
    axes[1].set_xlabel("Czas [s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Zoom na reprezentatywne okno 5 sekund (od 10.0 do 15.0 s), aby dokładnie pokazać kształt fali EKG
    axes[0].set_xlim(10.0, min(15.0, t[-1]))

    plt.tight_layout()
    if save and record_name:
        os.makedirs("results", exist_ok=True)
        fname = os.path.join("results", f"reconstruction_quality_{record_name}_{sample_idx}.png")
        plt.savefig(fname, dpi=150)
        print(f"  Zapisano: {fname}")
    plt.show()


def plot_hrv_comparison(t, ecg_full, disp_gt_peaks, scg_plot_norm, disp_pred_peaks, record_name, fs=256):
    """Trzy subploty: GT ECG z pikami, SCG z predykcjami, synchronizacja pików."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # --- Panel 1: EKG ground truth z pikami ---
    axes[0].plot(t, ecg_full, color='red', label='Referencyjne EKG')
    if len(disp_gt_peaks) > 0:
        local_idx = (disp_gt_peaks - int(t[0] * fs)).astype(int)
        valid = (local_idx >= 0) & (local_idx < len(ecg_full))
        axes[0].plot(disp_gt_peaks[valid] / float(fs),
                     ecg_full[local_idx[valid]], 'rx', markersize=10, label='GT R-Peaks')
    else:
        print("  [Ostrzeżenie] Brak GT peaks w oknie wizualizacji.")
    axes[0].set_title('Referencja (Ground Truth)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: SCG z predykcjami modelu ---
    axes[1].plot(t, scg_plot_norm, color='blue', alpha=0.6, label='SCG (Z-score)')
    if len(disp_pred_peaks) > 0:
        local_idx = (disp_pred_peaks - int(t[0] * fs)).astype(int)
        valid = (local_idx >= 0) & (local_idx < len(scg_plot_norm))
        axes[1].plot(disp_pred_peaks[valid] / float(fs),
                     scg_plot_norm[local_idx[valid]], 'gx', markersize=10, label='Predykcje modelu')
    else:
        print("  [Ostrzeżenie] Brak predicted peaks w oknie wizualizacji.")
    axes[1].set_title('Detekcja modelu HR na sygnale SCG')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- Panel 3: Porównanie synchronizacji ---
    axes[2].plot(t, ecg_full, color='gray', alpha=0.3, label='EKG (tło)')
    if len(disp_gt_peaks) > 0:
        axes[2].vlines(disp_gt_peaks / float(fs), -2, 2, colors='r', linestyles='--', label='GT Peaks')
    if len(disp_pred_peaks) > 0:
        axes[2].vlines(disp_pred_peaks / float(fs), -2, 2, colors='g', linestyles='-', label='Pred Peaks')
    axes[2].set_title('Synchronizacja: GT vs Predykcja')
    axes[2].set_xlabel('Czas [s]')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    fname = os.path.join("results", f"hrv_eval_{record_name}.png")
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Zapisano: {fname}")


def plot_poincare(r_peaks, fs=256, record_name='', save=True):
    """
    Wykres Poincaré: RR[n] vs RR[n+1].
    Elipsa z SD1 (krótka oś) i SD2 (długa oś) wizualizuje zmienność rytmu.
    """
    r_peaks = np.asarray(r_peaks)
    if len(r_peaks) < 3:
        print("  [Poincaré] Za mało pików do wykresu.")
        return

    rr_ms = np.diff(r_peaks) / fs * 1000.0
    valid = (rr_ms > 300) & (rr_ms < 2000)
    rr = rr_ms[valid]
    if len(rr) < 3:
        return

    rr_n  = rr[:-1]
    rr_n1 = rr[1:]

    diff_rr = np.diff(rr)
    sd1 = np.sqrt(0.5) * np.std(diff_rr, ddof=1)
    sd2_sq = max(0.0, 2.0 * np.std(rr, ddof=1) ** 2 - 0.5 * np.std(diff_rr, ddof=1) ** 2)
    sd2 = np.sqrt(sd2_sq)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(rr_n, rr_n1, s=10, alpha=0.5, color='steelblue', label='Punkty RR')
    ax.set_xlabel("RR[n] [ms]")
    ax.set_ylabel("RR[n+1] [ms]")
    title = f"Wykres Poincaré  {record_name}" if record_name else "Wykres Poincaré"
    ax.set_title(f"{title}\nSD1 = {sd1:.1f} ms  |  SD2 = {sd2:.1f} ms")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Elipsa SD1/SD2 (orientacja 45°)
    theta = np.linspace(0, 2 * np.pi, 200)
    cx, cy = np.mean(rr_n), np.mean(rr_n1)
    angle = np.pi / 4
    ellipse_x = cx + sd2 * np.cos(theta) * np.cos(angle) - sd1 * np.sin(theta) * np.sin(angle)
    ellipse_y = cy + sd2 * np.cos(theta) * np.sin(angle) + sd1 * np.sin(theta) * np.cos(angle)
    ax.plot(ellipse_x, ellipse_y, 'r-', linewidth=2, label='Elipsa SD1/SD2')
    ax.legend()

    plt.tight_layout()
    if save and record_name:
        os.makedirs("results", exist_ok=True)
        fname = os.path.join("results", f"poincare_{record_name}.png")
        plt.savefig(fname, dpi=150)
        print(f"  Zapisano: {fname}")
    plt.show()


def plot_hrv_spectrum(r_peaks, fs=256, record_name='', save=True):
    """
    Gęstość widmowa mocy (PSD) sygnału RR z zaznaczonymi pasmami VLF/LF/HF.
    """
    r_peaks = np.asarray(r_peaks)
    if len(r_peaks) < 8:
        print("  [PSD] Za mało pików do analizy częstotliwościowej.")
        return

    rr_ms = np.diff(r_peaks) / fs * 1000.0
    valid = (rr_ms > 300) & (rr_ms < 2000)
    rr = rr_ms[valid]
    if len(rr) < 8:
        return

    fs_rr = 4.0
    t_rr = np.cumsum(rr) / 1000.0
    t_rr -= t_rr[0]
    t_interp = np.arange(0, t_rr[-1], 1.0 / fs_rr)
    rr_interp = np.interp(t_interp, t_rr, rr)

    nperseg = min(256, len(rr_interp))
    freqs, psd = welch(rr_interp, fs=fs_rr, nperseg=nperseg)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(freqs, psd, color='navy', linewidth=1.5)

    # Wypełnienie pasm
    ax.fill_between(freqs, psd, where=(freqs >= 0.0033) & (freqs < 0.04),
                    alpha=0.3, color='green',  label='VLF (0.0033–0.04 Hz)')
    ax.fill_between(freqs, psd, where=(freqs >= 0.04)   & (freqs < 0.15),
                    alpha=0.3, color='orange', label='LF  (0.04–0.15 Hz)')
    ax.fill_between(freqs, psd, where=(freqs >= 0.15)   & (freqs < 0.40),
                    alpha=0.3, color='red',    label='HF  (0.15–0.40 Hz)')

    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Częstotliwość [Hz]")
    ax.set_ylabel("PSD [ms²/Hz]")
    title = f"Widmo HRV (Welch PSD)  {record_name}" if record_name else "Widmo HRV (Welch PSD)"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save and record_name:
        os.makedirs("results", exist_ok=True)
        fname = os.path.join("results", f"hrv_spectrum_{record_name}.png")
        plt.savefig(fname, dpi=150)
        print(f"  Zapisano: {fname}")
    plt.show()

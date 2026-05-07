import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Zapewnienie, że importy działają z poziomu głównego katalogu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import DataLoader
from preprocessing import Preprocessor

def run_preprocessing_tests():
    base_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    loader = DataLoader(base_data_dir=base_data)
    
    results_dir = os.path.join(os.path.dirname(__file__), "results_test_preprocessing")
    os.makedirs(results_dir, exist_ok=True)
    
    summary_lines = []
    summary_lines.append("=== PODSUMOWANIE TESTÓW PREPROCESSINGU ===\n")
    
    def test_preprocessor(name, list_func, load_func, default_fs=256, records_to_visualize=2):
        print(f"--- Testowanie preprocessingu dla {name} ---")
        records = list_func()
        summary_lines.append(f"{name}: Znaleziono {len(records)} rekordów ogółem.")
        
        success = 0
        total_epochs = 0
        rejected_epochs = 0
        retained_epochs = 0
        all_retained_stds = []
        
        for i, record in enumerate(records):
            df = load_func(record=record, format=True)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                # Wykryj fs
                fs = getattr(df, 'attrs', {}).get('fs', default_fs)
                if name == "IEEE":
                    fs = 800 # Zgodnie z formatem
                print(f"[{record}] Przetwarzanie (fs={fs})...")
                
                prep = Preprocessor(fs=fs)
                results = prep.process_pipeline(df)
                
                if results is not None:
                    n_env = len(results.get('peaks_env', []))
                    n_morph = len(results.get('peaks_morph', []))
                    print(f"[{record}] Preprocessing udany. Wykryto uderzeń: Envelope={n_env}, Morphological={n_morph}")
                    success += 1
                    
                    # Analiza epok
                    clean_mask = results.get('clean_mask', [])
                    epoch_sec = results.get('epoch_sec', 10)
                    scg_raw = results.get('scg_raw')
                    
                    if len(clean_mask) > 0 and scg_raw is not None:
                        n_epochs = len(clean_mask)
                        n_rejected = int(np.sum(~clean_mask))
                        n_retained = int(np.sum(clean_mask))
                        
                        total_epochs += n_epochs
                        rejected_epochs += n_rejected
                        retained_epochs += n_retained
                        
                        n_samples_epoch = int(epoch_sec * fs)
                        for ep_idx, is_clean in enumerate(clean_mask):
                            if is_clean:
                                ep_data = scg_raw[ep_idx*n_samples_epoch : (ep_idx+1)*n_samples_epoch]
                                if len(ep_data) > 0:
                                    all_retained_stds.append(np.std(ep_data))
                    
                    # Wizualizacja procesu (tylko dla określonej liczby rekordów)
                    if i < records_to_visualize:
                        samples_to_plot = int(10 * fs) # 10 sekund dla lepszego oglądu
                        
                        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                        fig.suptitle(f"Preprocessing {name} - {record}")
                        
                        time_axis = np.arange(samples_to_plot) / fs
                        
                        # Ogranicz dane do plotu
                        scg_f = results['scg_f']
                        scg_d = results['scg_d']
                        scg_k = results['scg_kaisti']
                        
                        plot_len = min(samples_to_plot, len(scg_raw))
                        time_axis = time_axis[:plot_len]
                        
                        axes[0].plot(time_axis, scg_raw[:plot_len], label='Raw SCG', color='#1f77b4')
                        axes[0].set_ylabel('Amplitude')
                        axes[0].legend(loc='upper right')
                        
                        axes[1].plot(time_axis, scg_f[:plot_len], label='Filtered (0.5-20Hz)', color='#ff7f0e')
                        axes[1].set_ylabel('Amplitude')
                        axes[1].legend(loc='upper right')
                        
                        axes[2].plot(time_axis, scg_d[:plot_len], label='Differentiated', color='#2ca02c')
                        axes[2].set_ylabel('Amplitude')
                        axes[2].legend(loc='upper right')
                        
                        axes[3].plot(time_axis, scg_k[:plot_len], label='Kaisti Normalized', color='#9467bd')
                        
                        peaks_env = results['peaks_env']
                        peaks_in_window = peaks_env[peaks_env < plot_len]
                        if len(peaks_in_window) > 0:
                            axes[3].scatter(time_axis[peaks_in_window], scg_k[peaks_in_window], color='red', marker='x', s=100, label='Peaks (Env)')
                            
                        axes[3].set_xlabel('Time [s]')
                        axes[3].set_ylabel('Z-Score')
                        axes[3].legend(loc='upper right')
                        
                        plt.tight_layout()
                        save_path = os.path.join(results_dir, f"{name.lower()}_{record}_prep.png")
                        plt.savefig(save_path, dpi=150)
                        plt.close()
                        print(f"[{record}] Zapisano wykres do {save_path}")
                else:
                    print(f"[{record}] Preprocessing zwrócił None (Brak odpowiednich kolumn?)")
            else:
                print(f"[{record}] Błąd ładowania danych")
                
        # Podsumowanie dla danego datasetu
        summary_lines.append(f" - Sukces ładowania: {success}/{len(records)}")
        summary_lines.append(f" - Suma epok: {total_epochs}")
        if total_epochs > 0:
            summary_lines.append(f" - Odrzucone epoki: {rejected_epochs} ({rejected_epochs/total_epochs*100:.2f}%)")
            summary_lines.append(f" - Pozostawione epoki: {retained_epochs} ({retained_epochs/total_epochs*100:.2f}%)")
        else:
            summary_lines.append(f" - Odrzucone epoki: 0 (0%)")
            summary_lines.append(f" - Pozostawione epoki: 0 (0%)")
            
        avg_std = np.mean(all_retained_stds) if len(all_retained_stds) > 0 else 0
        summary_lines.append(f" - Średnie odchylenie std pozostawionych epok (SCG): {avg_std:.4f}\n")

    # Uruchomienie testów dla każdej z kolekcji
    test_preprocessor("IEEE", loader.list_ieee, loader.load_ieee, default_fs=800, records_to_visualize=2)
    test_preprocessor("Zenodo", loader.list_zenodo, loader.load_zenodo, default_fs=256, records_to_visualize=2)
    test_preprocessor("PhysioNet", loader.list_physionet, loader.load_physionet, default_fs=256, records_to_visualize=2)
    
    # Zapis logów numerycznych
    summary_path = os.path.join(results_dir, "preprocessing_summary.txt")
    with open(summary_path, "w", encoding='utf-8') as f:
        f.writelines([line + "\n" for line in summary_lines])
    print(f"Zapisano podsumowanie testów do {summary_path}")

if __name__ == "__main__":
    run_preprocessing_tests()

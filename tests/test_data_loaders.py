import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# Ustaw wartość na liczbę całkowitą (np. 42), aby uzyskać powtarzalne wyniki losowania, 
# lub pozostaw None dla pełnej losowości przy każdym uruchomieniu.
RANDOM_SEED = None

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

# Zapewnienie, że importy działają z poziomu głównego katalogu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import DataLoader

def run_tests():
    # Używamy ścieżki do folderu data z perspektywy katalogu głównego
    base_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    loader = DataLoader(base_data_dir=base_data)
    
    results_dir = os.path.join(os.path.dirname(__file__), "results_test_data_loaders")
    os.makedirs(results_dir, exist_ok=True)
    
    summary_lines = []
    summary_lines.append("=== PODSUMOWANIE TESTÓW DATA LOADERÓW ===\n")
    
    def test_dataset(name, list_func, load_func, default_cols):
        print(f"--- Testowanie {name} ---")
        records = list_func()
        print(f"Znaleziono {len(records)} rekordów {name}: {records[:5]}")
        summary_lines.append(f"{name}: Znaleziono {len(records)} rekordów ogółem.")
        
        success = 0
        tested = min(3, len(records))
        chosen_records = random.sample(records, tested) if len(records) > tested else records
        for i, record in enumerate(chosen_records):
            df = load_func(record=record, format=True)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                print(f"Rekord {record} załadowany poprawnie. Ksztalt: {df.shape}, Kolumny: {list(df.columns)}")
                success += 1
                
                # Plot
                plt.figure(figsize=(12, 6))
                
                cols_to_plot = [c for c in default_cols if c in df.columns]
                if not cols_to_plot:
                    cols_to_plot = list(df.columns)[:3]
                
                # Losowanie okna czasowego (stała długość 2000)
                window_length = 2000
                max_start = max(0, len(df) - window_length)
                start_idx = random.randint(0, max_start)
                end_idx = start_idx + window_length
                
                for col in cols_to_plot:
                    # Rysujemy 2000 próbek z wylosowanego okna
                    plt.plot(df[col].values[start_idx:end_idx], label=col, alpha=0.8)
                    
                plt.title(f"{name} - {record} (Samples: {start_idx}-{end_idx})")
                plt.xlabel("Sample Index")
                plt.ylabel("Amplitude")
                plt.legend(loc='upper right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                save_path = os.path.join(results_dir, f"{name.lower()}_{record}.png")
                plt.savefig(save_path, dpi=150)
                plt.close()
                print(f"Zapisano wykres do: {save_path}")
            else:
                print(f"Błąd ładowania rekordu {record} dla {name}")
                
        print(f"Pomyślnie załadowano {success}/{tested} rekordów {name} do wizualizacji.\n")
        summary_lines.append(f" - Przetestowano {tested} rekordów: sukces {success}/{tested}.\n")

    # Uruchomienie testów dla każdej z kolekcji
    test_dataset("IEEE", loader.list_ieee, loader.load_ieee, ['ECG_LA_RA', 'SCG_Z', 'GCG_Y', 'EKG', 'accZ', 'gyroY'])
    test_dataset("Zenodo", loader.list_zenodo, loader.load_zenodo, ['ECG_LA_RA', 'SCG_Z', 'GCG_Y'])
    test_dataset("PhysioNet", loader.list_physionet, loader.load_physionet, ['ECG', 'SCG', 'RESP'])
    
    # Zapis logów numerycznych
    summary_path = os.path.join(results_dir, "test_summary.txt")
    with open(summary_path, "w", encoding='utf-8') as f:
        f.writelines([line + "\n" for line in summary_lines])
    print(f"Zapisano podsumowanie testów do {summary_path}")

if __name__ == "__main__":
    run_tests()

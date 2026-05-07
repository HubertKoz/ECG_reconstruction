import os
import pandas as pd
import wfdb

class PhysioNetMixin:
    # Listowanie
    def list_physionet(self, path: str ='PhysioNet') -> list[str]:
        semi_prefix = os.path.join(self.base_data_dir, path)
        records = sorted([f[:-4] for f in os.listdir(semi_prefix) if f.endswith('.hea')])
        return records

    # Loader
    def load_physionet(self, path: str ='PhysioNet', record: str ='b001', format=False) -> tuple[wfdb.Record, wfdb.Annotation | None] | pd.DataFrame | None:
        """
        Zbiór: Combined measurement of ECG, Breathing and Seismocardiograms (PhysioNet)
        Format: Standard WFDB obsługiwany przez bibliotekę wfdb dla Pythona.
        """
        full_path = os.path.join(self.base_data_dir, path, record)
        print(f"[PhysioNet] Ładowanie WFDB z {full_path}...")
        
        try:
            # Wczytywanie surowych sygnałów z pliku .dat oraz definicji z pliku .hea
            record = wfdb.rdrecord(full_path)
            
            # Wiele rekordów w PhysioNet zawiera adnotacje np. w pliku .atr (adnotacje uderzeń r)
            # Spróbujmy wczytać referencyjne anotatory jeśli istnieją
            try:
                annotation = wfdb.rdann(full_path, 'atr')
            except FileNotFoundError:
                annotation = None
                
            if format:
                return self.physionet_adapter((record, annotation))
            return record, annotation
        except Exception as e:
            print(f"Błąd podczas ładowania danych PhysioNet: {e}")
            return None, None


    # Adapter
    def physionet_adapter(self, record_tuple: tuple, fs=5000) -> pd.DataFrame:
        """
        Adapter konwertujący wynik load_physionet_wfdb (Record) na pd.DataFrame.
        """
        record, _ = record_tuple
        if record is None:
            return pd.DataFrame()
        
        df = pd.DataFrame(record.p_signal, columns=record.sig_name)
        return self.resample(df, fs, 256)

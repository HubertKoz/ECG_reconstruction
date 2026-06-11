import os
import pandas as pd
import wfdb

from config import TARGET_FS


class PhysioNetMixin:

    def list_physionet(self, path: str = 'PhysioNet') -> list[str]:
        prefix = os.path.join(self.base_data_dir, path)
        return sorted(f[:-4] for f in os.listdir(prefix) if f.endswith('.hea'))

    def load_physionet(
        self,
        path: str = 'PhysioNet',
        record: str = 'b001',
        format: bool = False,
    ) -> tuple[wfdb.Record, wfdb.Annotation | None] | pd.DataFrame | None:
        """
        Wczytuje rekord PhysioNet CEBS (format WFDB).
        Zwraca (wfdb.Record, wfdb.Annotation | None) lub DataFrame po format=True.
        Adnotacje .atr zawierają ręcznie oznaczone szczyty R.
        """
        full_path = os.path.join(self.base_data_dir, path, record)
        print(f"[PhysioNet] Ładowanie WFDB z {full_path}...")

        try:
            rec = wfdb.rdrecord(full_path)
            try:
                annotation = wfdb.rdann(full_path, 'atr')
            except FileNotFoundError:
                annotation = None

            if format:
                return self.physionet_adapter((rec, annotation))
            return rec, annotation

        except Exception as e:
            print(f"[PhysioNet] Błąd podczas ładowania {record}: {e}")
            return None, None

    def physionet_adapter(self, record_tuple: tuple) -> pd.DataFrame:
        """
        Konwertuje (wfdb.Record, _) na pd.DataFrame i resampluje do TARGET_FS.
        Częstotliwość próbkowania odczytywana z metadanych WFDB (domyślnie 5000 Hz).
        """
        rec, _ = record_tuple
        if rec is None:
            return pd.DataFrame()

        fs = int(rec.fs) if hasattr(rec, 'fs') and rec.fs else 5000
        df = pd.DataFrame(rec.p_signal, columns=rec.sig_name)
        return self.resample(df, original_fs=fs)

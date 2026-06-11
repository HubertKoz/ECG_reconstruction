import os
import pandas as pd

from config import TARGET_FS


# Mapowanie nazw kolumn IEEE → standard Zenodo (spójność między zbiorami).
# EKG  — sub_1–7  (starszy chip F-EKG V.3)
# ECG  — sub_8–29 (TI ADS1293); ECG1/ECG2/NC są pomijane (nie mają odpowiednika)
_IEEE_TO_ZENODO = {
    'EKG':   'ECG_LA_RA',
    'ECG':   'ECG_LA_RA',
    'accX':  'SCG_X',
    'accY':  'SCG_Y',
    'accZ':  'SCG_Z',
    'gyroX': 'GCG_X',
    'gyroY': 'GCG_Y',
    'gyroZ': 'GCG_Z',
}

_IEEE_RAW_FS = 800  # Hz — stała dla wszystkich rekordów IEEE


class IEEEMixin:

    def list_ieee(self, path: str = 'IEEE') -> list[str]:
        prefix = os.path.join(self.base_data_dir, path)
        return sorted(f[:-4] for f in os.listdir(prefix) if f.endswith('.txt') and f.startswith('sub_'))

    def load_ieee(self, path: str = 'IEEE', record: str = 'sub_1', format: bool = False) -> pd.DataFrame | None:
        """
        Wczytuje rekord IEEE DataPort (.txt z sekcjami [HEADER]/[SENSORS]/[DATA]).
        Zwraca DataFrame ze zunifikowanymi nazwami kolumn (standard Zenodo).
        format=True → resampling do TARGET_FS.
        """
        full_path = os.path.join(self.base_data_dir, path, record + '.txt')
        if not os.path.exists(full_path):
            print(f"[IEEE] Błąd: brak pliku {full_path}")
            return None

        print(f"[IEEE] Ładowanie {full_path}...")

        try:
            sensor_names: list[str] = []
            data_start_line = 0
            section = None

            with open(full_path, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped == '[HEADER]':
                        section = 'HEADER'
                    elif stripped == '[SENSORS]':
                        section = 'SENSORS'
                    elif stripped == '[DATA]':
                        data_start_line = i + 1
                        break
                    elif section == 'SENSORS' and stripped.startswith('Signal'):
                        try:
                            info = stripped.split(':', 1)[1].strip()
                            sensor_names.append(info.split(',')[0].strip())
                        except IndexError:
                            continue

            df = pd.read_csv(
                full_path,
                sep=r'\s+',
                header=None,
                skiprows=data_start_line,
                names=sensor_names,
            )
            df.rename(columns={k: v for k, v in _IEEE_TO_ZENODO.items() if k in df.columns}, inplace=True)

            if format:
                return self.ieee_adapter(df)
            return df

        except Exception as e:
            print(f"[IEEE] Błąd podczas ładowania {record}: {e}")
            return None

    def ieee_adapter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resampluje z _IEEE_RAW_FS do TARGET_FS."""
        return self.resample(df, original_fs=_IEEE_RAW_FS)

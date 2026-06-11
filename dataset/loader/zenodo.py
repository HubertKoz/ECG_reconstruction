import os
import pandas as pd

from config import TARGET_FS


# Mapowanie oryginalnych nazw kolumn Shimmer → zunifikowane nazwy pipeline.
# CP records używają prefiksu "Shimmer_D0CD_", UP records używają "ECG_".
_ZENODO_RENAME = {
    # CP — prefix Shimmer_D0CD_
    'Shimmer_D0CD_Timestamp_Shimmer_CAL':  'Timestamp',
    'Shimmer_D0CD_Accel_LN_X_CAL':         'SCG_X',
    'Shimmer_D0CD_Accel_LN_Y_CAL':         'SCG_Y',
    'Shimmer_D0CD_Accel_LN_Z_CAL':         'SCG_Z',
    'Shimmer_D0CD_ECG_LA-RA_24BIT_CAL':    'ECG_LA_RA',
    'Shimmer_D0CD_ECG_LL-LA_24BIT_CAL':    'ECG_LL_LA',
    'Shimmer_D0CD_Gyro_X_CAL':             'GCG_X',
    'Shimmer_D0CD_Gyro_Y_CAL':             'GCG_Y',
    'Shimmer_D0CD_Gyro_Z_CAL':             'GCG_Z',
    # UP — prefix ECG_
    'ECG_Timestamp_Unix_CAL':              'Timestamp',
    'ECG_TimestampSync_Unix_CAL':          'Timestamp',
    'ECG_Accel_LN_X_CAL':                  'SCG_X',
    'ECG_Accel_LN_Y_CAL':                  'SCG_Y',
    'ECG_Accel_LN_Z_CAL':                  'SCG_Z',
    'ECG_ECG_LA-RA_24BIT_CAL':             'ECG_LA_RA',
    'ECG_ECG_LL-LA_24BIT_CAL':             'ECG_LL_LA',
    'ECG_Gyro_X_CAL':                      'GCG_X',
    'ECG_Gyro_Y_CAL':                      'GCG_Y',
    'ECG_Gyro_Z_CAL':                      'GCG_Z',
}

# Kolumny sygnałowe przekazywane dalej przez pipeline
# (Timestamp jest pomocniczy — tylko do detekcji fs)
_DESIRED_COLS = ['SCG_X', 'SCG_Y', 'SCG_Z', 'ECG_LA_RA', 'ECG_LL_LA', 'GCG_X', 'GCG_Y', 'GCG_Z']

# Standardowe fs urządzeń Shimmer — do zaokrąglenia zmierzonej wartości
_STANDARD_FS = [51, 102, 128, 204, 256, 512, 1024]


class ZenodoMixin:

    def list_zenodo(self, path: str = 'Zenodo/Raw_Recordings') -> list[str]:
        prefix = os.path.join(self.base_data_dir, path)
        return sorted(set(f[:-8] for f in os.listdir(prefix) if f.endswith('-Raw.csv')))

    def load_zenodo(self, path: str = 'Zenodo/Raw_Recordings', record: str = 'CP-01', format: bool = False) -> pd.DataFrame | None:
        """
        Wczytuje rekord Zenodo VHD (plik CSV z Shimmer 3 ECG).

        Obsługiwane formaty pliku:
          Format A — linia 0: "sep=<sep>", linia 1: nagłówki, linia 2: jednostki (pomijana)
          Format B — linia 0: nagłówki, linia 1: jednostki (pomijana)

        Korekta osi Y dla CP-01 do CP-38 (sensor zamontowany odwrotnie).
        format=True → resampling do TARGET_FS.
        """
        full_path = os.path.join(self.base_data_dir, path, record + '-Raw.csv')
        print(f"[Zenodo] Ładowanie {full_path}...")

        try:
            with open(full_path, encoding='utf-8') as f:
                first_line = f.readline().strip()

            clean = first_line.lower().strip('"').strip("'")
            if clean.startswith('sep='):
                sep_val = clean.split('=', 1)[1]
                separator = '\t' if sep_val in ('\\t', '\t') else (sep_val or ',')
                skip_rows = [0, 2]   # pominięcie linii sep= oraz linii jednostek
            else:
                separator = ','
                skip_rows = [1]      # pominięcie wyłącznie linii jednostek

            df = pd.read_csv(full_path, sep=separator, skiprows=skip_rows, header=0, engine='python')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.rename(columns={k: v for k, v in _ZENODO_RENAME.items() if k in df.columns}, inplace=True)

            # Korekta osi Y dla CP-01 do CP-38 (sensor zamontowany odwrotnie)
            if record.startswith('CP-'):
                try:
                    rec_num = int(record.split('-', 1)[1])
                    if 1 <= rec_num <= 38:
                        print(f"   -> Odwracanie osi Y dla {record} (CP-01–CP-38)")
                        if 'SCG_Y' in df.columns:
                            df['SCG_Y'] = -df['SCG_Y']
                        if 'GCG_Y' in df.columns:
                            df['GCG_Y'] = -df['GCG_Y']
                except ValueError:
                    pass

            # Detekcja częstotliwości próbkowania z timestampów
            fs = TARGET_FS
            if 'Timestamp' in df.columns:
                diffs = df['Timestamp'].diff().dropna()
                avg_diff_ms = diffs.median()
                if avg_diff_ms > 0:
                    raw_fs = 1000.0 / avg_diff_ms
                    fs = min(_STANDARD_FS, key=lambda x: abs(x - raw_fs))
                    print(f"   -> Wykryte fs dla {record}: {fs} Hz (raw={raw_fs:.1f} Hz)")
                else:
                    print(f"   -> Ostrzeżenie: delta Timestamp = {avg_diff_ms} ms — zakładam {TARGET_FS} Hz")
            else:
                print(f"   -> Brak kolumny Timestamp — zakładam {TARGET_FS} Hz")

            target_cols = [c for c in _DESIRED_COLS if c in df.columns]

            if format:
                return self.zenodo_adapter(df[target_cols], fs=fs)
            return df[target_cols]

        except Exception as e:
            print(f"[Zenodo] Błąd podczas ładowania {record}: {e}")
            return None

    def zenodo_adapter(self, data: pd.DataFrame, fs: int = TARGET_FS) -> pd.DataFrame:
        """Resampluje do TARGET_FS (niezależnie od wykrytego fs)."""
        return self.resample(data, original_fs=fs)

    def load_zenodo_json_peaks(self, path: str = 'Zenodo/JSON_Files', record: str = 'CP-01') -> dict | None:
        """
        Wczytuje adnotacje szczytów R z plików JSON (Zenodo).
        Zwraca słownik z listami czasów dla poszczególnych odprowadzeń.
        """
        import json
        full_path = os.path.join(self.base_data_dir, path, record + '-ECG.json')
        if not os.path.exists(full_path):
            full_path = os.path.join(self.base_data_dir, path, record + '.json')
            if not os.path.exists(full_path):
                print(f"[Zenodo] Brak pliku adnotacji: {full_path}")
                return None
        try:
            with open(full_path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Zenodo] Błąd podczas ładowania JSON {record}: {e}")
            return None

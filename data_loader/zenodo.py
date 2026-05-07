import os
import pandas as pd

class ZenodoMixin:
    # Listowanie
    def list_zenodo(self, path: str ='Zenodo/Raw_Recordings') -> list[str]:
        semi_prefix = os.path.join(self.base_data_dir, path)
        records = sorted(list(set([f[:-8] for f in os.listdir(semi_prefix) if f.endswith('-Raw.csv')])))  
        return records

    def load_zenodo(self, path: str ='Zenodo/Raw_Recordings', record: str = 'CP-01', format=False) -> pd.DataFrame | None:
        """
        Zbiór: Surowe dane z urządzenia Shimmer (Zenodo/VHD)
        Struktura może się różnić:
        Wersja A:
        Lp 1: Definicja separatora (np. 'sep=	')
        Lp 2: Nazwy kolumn (Headers)
        Lp 3: Jednostki (Units)
        Wersja B:
        Lp 1: Nazwy kolumn (Headers)
        Lp 2: Jednostki (Units)
        """
        full_path = os.path.join(self.base_data_dir, path, record + '-Raw.csv')
        print(f"[Shimmer CSV] Ładowanie danych z {full_path}...")

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
            separator = ','
            skip_rows = []
            
            clean_first_line = first_line.lower().strip('"').strip("'")
            if clean_first_line.startswith('sep='):
                val = clean_first_line.split('=')[1]
                if val in ['\\t', '\t']:
                    separator = '\t'
                elif val == ',':
                    separator = ','
                else: 
                    separator = val
                skip_rows = [0, 2]
            else:
                skip_rows = [1]
            
            df = pd.read_csv(
                full_path, 
                sep=separator, 
                skiprows=skip_rows, 
                header=0,
                engine='python'
            )

            # Usunięcie ew. pustych kolumn na końcu (częsty błąd z białymi znakami jako separatorem)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            rename_map = {
                'Shimmer_D0CD_Timestamp_Shimmer_CAL': 'Timestamp',
                'Shimmer_D0CD_Accel_LN_X_CAL': 'SCG_X',
                'Shimmer_D0CD_Accel_LN_Y_CAL': 'SCG_Y',
                'Shimmer_D0CD_Accel_LN_Z_CAL': 'SCG_Z',
                'Shimmer_D0CD_ECG_LA-RA_24BIT_CAL': 'ECG_LA_RA',
                'Shimmer_D0CD_ECG_LL-LA_24BIT_CAL': 'ECG_LL_LA',
                'Shimmer_D0CD_Gyro_X_CAL': 'GCG_X',
                'Shimmer_D0CD_Gyro_Y_CAL': 'GCG_Y',
                'Shimmer_D0CD_Gyro_Z_CAL': 'GCG_Z',
                
                # Dodatkowe nazwy występujące w plikach UP
                'ECG_Timestamp_Unix_CAL': 'Timestamp',
                'ECG_TimestampSync_Unix_CAL': 'Timestamp',
                'ECG_Accel_LN_X_CAL': 'SCG_X',
                'ECG_Accel_LN_Y_CAL': 'SCG_Y',
                'ECG_Accel_LN_Z_CAL': 'SCG_Z',
                'ECG_ECG_LA-RA_24BIT_CAL': 'ECG_LA_RA',
                'ECG_ECG_LL-LA_24BIT_CAL': 'ECG_LL_LA',
                'ECG_Gyro_X_CAL': 'GCG_X',
                'ECG_Gyro_Y_CAL': 'GCG_Y',
                'ECG_Gyro_Z_CAL': 'GCG_Z'
            }
            
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

            if record.startswith('CP-'):
                try:
                    rec_num = int(record.split('-')[1])
                    if 1 <= rec_num <= 38:
                        print(f"   -> Info: Odwracanie osi Y dla SCG i GCG (dla CP01-38).")
                        if 'SCG_Y' in df.columns:
                            df['SCG_Y'] = -df['SCG_Y']
                        if 'GCG_Y' in df.columns:
                            df['GCG_Y'] = -df['GCG_Y']
                except ValueError:
                    pass

            fs = 256
            if 'Timestamp' in df.columns:
                diffs = df['Timestamp'].diff().dropna()
                avg_diff_ms = diffs.median()
                if avg_diff_ms > 0:
                    fs = round(1000.0 / avg_diff_ms)
                    print(f"   -> Wykryte próbkowanie dla {record}: {fs} Hz (Delta t: {avg_diff_ms:.2f} ms)")
                else:
                    print(f"   -> Ostrzeżenie: Wykryta delta Timestamp wyniosła {avg_diff_ms} ms dla {record}. Zakładam domyślne fs = 256 Hz.")
                df.attrs['fs'] = fs
            else:
                print(f"   -> Ostrzeżenie: Brak kolumny Timestamp w {record}. Nie można wyliczyć fs. Zakładam 256 Hz.")

            desired_cols = ['SCG_X', 'SCG_Y', 'SCG_Z', 'ECG_LA_RA', 'ECG_LL_LA', 'GCG_X', 'GCG_Y', 'GCG_Z']
            target_cols = [c for c in desired_cols if c in df.columns]
            
            if format:
                return self.zenodo_adapter(df[target_cols], fs=fs)
            return df[target_cols]

        except Exception as e:
            print(f"Błąd podczas ładowania Shimmer CSV ({record}): {e}")
            return None

    def zenodo_adapter(self, data: pd.DataFrame, fs=800) -> pd.DataFrame:
        if fs < 800:
            return self.resample(data, fs)
        return data

    def load_zenodo_json_peaks(self, path: str ='Zenodo/JSON_Files', record: str = 'CP-01') -> dict | None:
        """
        Ładuje adnotacje szczytów R z plików JSON (Zenodo).
        Zwraca słownik z listami czasów (string) dla poszczególnych odprowadzeń.
        """
        import json
        full_path = os.path.join(self.base_data_dir, path, record + '-ECG.json')
        if not os.path.exists(full_path):
            # Próba znalezienia pliku bez przyrostka -ECG (niektóre mogą mieć inną nazwę)
            full_path = os.path.join(self.base_data_dir, path, record + '.json')
            if not os.path.exists(full_path):
                print(f"Błąd: Plik adnotacji {full_path} nie istnieje.")
                return None
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Błąd podczas ładowania JSON Zenodo: {e}")
            return None

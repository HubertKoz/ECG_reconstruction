import os
import pandas as pd

class IEEEMixin:
    # Listowanie
    def list_ieee(self, path: str ='IEEE') -> list[str]:
        semi_prefix = os.path.join(self.base_data_dir, path)
        records = sorted([f[:-4] for f in os.listdir(semi_prefix) if f.endswith('.txt') and f.startswith('sub_')])
        return records

    # Loader
    def load_ieee(self, path: str ='IEEE', record: str ='sub_1', format=False) -> pd.DataFrame | None:
        """
        Zbiór: Mechanocardiograms with ECG reference (IEEE DataPort)
        Format: Płaskie pliki tekstowe (.txt) z danymi 6-DoF oraz EKG.
        Dostosowany do obsługi tagów [HEADER], [SENSORS] i [DATA].
        """
        full_path = os.path.join(self.base_data_dir, path, record + '.txt')
        if not os.path.exists(full_path):
            print(f"Błąd: Plik {full_path} nie istnieje.")
            return None
            
        print(f"[IEEE DataPort] Ładowanie danych z {full_path}...")
        
        try:
            header_content = []
            sensor_names = []
            data_start_line = 0
            section = None
            
            # Przeszukiwanie pliku pod kątem tagów i metadanych
            with open(full_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    stripped = line.strip()
                    if not stripped:
                        continue
                        
                    if stripped == '[HEADER]':
                        section = 'HEADER'
                        continue
                    elif stripped == '[SENSORS]':
                        section = 'SENSORS'
                        continue
                    elif stripped == '[DATA]':
                        data_start_line = i + 1
                        break
                        
                    if section == 'HEADER':
                        header_content.append(stripped)
                    elif section == 'SENSORS':
                        # Przykładowa linia: "Signal1: EKG, F-EKG V.3 16072013"
                        if stripped.startswith('Signal'):
                            try:
                                # Wyciągamy to co po dwukropku, potem bierzemy to co przed pierwszym przecinkiem
                                info = stripped.split(':')[1].strip()
                                sensor_name = info.split(',')[0].strip()
                                sensor_names.append(sensor_name)
                            except IndexError:
                                continue

            # Wyświetlenie metadanych
            # print("Nagłówek pliku:")
            # for hl in header_content:
            #     print(f"  {hl}")
            
            # print(f"Wykryto {len(sensor_names)} sensorów: {', '.join(sensor_names)}")
            
            # Wczytanie danych (skiprows pomija wszystko do tagu [DATA] włącznie)
            df = pd.read_csv(full_path, sep=r'\s+', header=None, skiprows=data_start_line, names=sensor_names)
            
            # Mapowanie nazw na standard Zenodo dla spójności między zbiorami
            ieee_to_zenodo = {
                'EKG': 'ECG_LA_RA',
                'accX': 'SCG_X',
                'accY': 'SCG_Y',
                'accZ': 'SCG_Z',
                'gyroX': 'GCG_X',
                'gyroY': 'GCG_Y',
                'gyroZ': 'GCG_Z'
            }
            
            df.rename(columns=ieee_to_zenodo, inplace=True)
            if format:
                return self.ieee_adapter(df)
            return df
            
        except Exception as e:
            print(f"Błąd podczas ładowania danych IEEE: {e}")
            return None

    # Adapter
    def ieee_adapter(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.resample(df, 800)

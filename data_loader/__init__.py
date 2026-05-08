# Inicjalizacja pakietu data_loader i agregacja funkcjonalności
from .base import BaseLoader
from .ieee import IEEEMixin
from .zenodo import ZenodoMixin
from .physionet import PhysioNetMixin

class DataLoader(BaseLoader, IEEEMixin, ZenodoMixin, PhysioNetMixin):
    """
    Główna klasa ładująca wszystkie typy danych poprzez mechanizmy dziedziczenia klas pomocniczych.
    Zgodna w 100% z poprzednim wdrożeniem dla zachowania kompatybilności wstecznej wszystkich plików.
    """
    def __init__(self, base_data_dir: str = "./data"):
        super().__init__(base_data_dir)

    def load_all_datasets(self, format=True):
        """
        Pobiera i przygotowuje do dalszej obróbki wszystkie dostępne rekordy ze wszystkich zbiorów.
        Zwraca słownik, gdzie klucze to nazwy zbiorów, a wartości to listy ramek danych (pd.DataFrame).
        """
        print("Rozpoczynanie ładowania wszystkich zbiorów danych...")
        datasets = {
            'ieee': [],
            'zenodo': [],
            'physionet': []
        }
        
        # 1. IEEE DataPort
        print(f" -> Ładowanie IEEE ({len(self.list_ieee())} rekordów)")
        for record in self.list_ieee():
            df = self.load_ieee(record=record, format=format)
            if df is not None:
                datasets['ieee'].append(df)
        
        # 2. Zenodo / Shimmer
        print(f" -> Ładowanie Zenodo ({len(self.list_zenodo())} rekordów)")
        for record in self.list_zenodo():
            df = self.load_zenodo(record=record, format=format)
            if df is not None:
                datasets['zenodo'].append(df)
                
        # 3. PhysioNet
        print(f" -> Ładowanie PhysioNet ({len(self.list_physionet())} rekordów)")
        for record in self.list_physionet():
            # format=True dla PhysioNet zwraca DataFrame resampled do 256Hz
            df = self.load_physionet(record=record, format=format)
            if df is not None:
                datasets['physionet'].append(df)
                
        print("Ładowanie zakończone.")
        return datasets


if __name__ == "__main__":
    # Testowy punkt wejścia do weryfikacji architektury
    print("Inicjalizacja DataLoader")
    loader = DataLoader()
    
    print("Test:")
    print()
    
    # IEEE
    print("### IEEE ###")
    print(loader.list_ieee()[:5])
    # raw_ieee = loader.load_ieee()
    # df_ieee = loader.ieee_adapter(raw_ieee)
    # print(df_ieee.head())
    print()
    
    # Zenodo
    print("### Zenodo ###")
    print(loader.list_zenodo()[:5])
    # raw_zenodo = loader.load_zenodo()
    # df_zenodo = loader.zenodo_adapter(raw_zenodo)
    # print(df_zenodo.head())
    print()
    
    # PhysioNet
    print("### PhysioNet ###")
    print(loader.list_physionet()[:5])
    # raw_physio = loader.load_physionet()
    # df_physio = loader.physionet_adapter(raw_physio)
    # print(df_physio.head())
    print()

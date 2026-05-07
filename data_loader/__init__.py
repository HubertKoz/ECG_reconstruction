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

from .base import BaseLoader
from .ieee import IEEEMixin
from .zenodo import ZenodoMixin
from .physionet import PhysioNetMixin


class DataLoader(BaseLoader, IEEEMixin, ZenodoMixin, PhysioNetMixin):
    """Ładuje dane z IEEE DataPort, Zenodo VHD i PhysioNet CEBS."""

    def __init__(self, base_data_dir: str = "./data"):
        super().__init__(base_data_dir)

    def load_all_datasets(self, format: bool = True) -> dict:
        """
        Wczytuje wszystkie rekordy ze wszystkich zbiorów.
        Zwraca {nazwa_zbioru: [list of DataFrames]}.
        """
        print("Ładowanie wszystkich zbiorów danych...")
        datasets: dict[str, list] = {'ieee': [], 'zenodo': [], 'physionet': []}

        print(f" -> IEEE ({len(self.list_ieee())} rekordów)")
        for record in self.list_ieee():
            df = self.load_ieee(record=record, format=format)
            if df is not None:
                datasets['ieee'].append(df)

        print(f" -> Zenodo ({len(self.list_zenodo())} rekordów)")
        for record in self.list_zenodo():
            df = self.load_zenodo(record=record, format=format)
            if df is not None:
                datasets['zenodo'].append(df)

        print(f" -> PhysioNet ({len(self.list_physionet())} rekordów)")
        for record in self.list_physionet():
            df = self.load_physionet(record=record, format=format)
            if df is not None:
                datasets['physionet'].append(df)

        print("Ładowanie zakończone.")
        return datasets

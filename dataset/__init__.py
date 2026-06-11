"""
dataset — pakiet ładujący i wstępnie przetwarzający dane pomiarowe.

Struktura:
    dataset.loader       — klasy ładujące surowe pliki (IEEE, Zenodo, PhysioNet)
    dataset.preprocessor — filtry, detekcja artefaktów, normalizacja, selekcja osi

Skrócony import (najczęstszy przypadek użycia):
    from dataset import DataLoader, Preprocessor
"""
from .loader import DataLoader
from .preprocessor import Preprocessor

__all__ = ['DataLoader', 'Preprocessor']

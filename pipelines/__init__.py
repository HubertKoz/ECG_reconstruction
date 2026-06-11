"""
pipelines — Funkcje do przetwarzania i agregacji danych SCG/GCG -> EKG.

Pakiet jest niezalezny od dataset/ — przyjmuje gotowe DataFramy i zwraca tablice numpy.

Importy jednej linii:
    from pipelines import kaisti_pipeline, PIPELINE_REGISTRY
    from pipelines import aggregate_balanced_sources
"""

# Pipeline'y sygnalowe
from .signal import (
    kaisti_pipeline,
    robust_pipeline,
    minimal_pipeline,
    wavelet_pipeline,
    subband_pipeline,
    corrected_pipeline,
    pca_pipeline,
)

# robust_pipeline jako alias za advanced_filtering_pipeline (stara nazwa)
advanced_filtering_pipeline = robust_pipeline

# Slownik pipeline'ow alternatywnych (wszystkie poza kaisti/advanced)
ALTERNATIVE_PIPELINES = {
    'robust':    robust_pipeline,
    'minimal':   minimal_pipeline,
    'wavelet':   wavelet_pipeline,
    'subband':   subband_pipeline,
    'corrected': corrected_pipeline,
    'pca':       pca_pipeline,
}

# Agregacja / balansowanie
from .orchestrate import aggregate_and_balance_datasets, aggregate_balanced_sources

# Rejestr wszystkich pipeline'ow (do dynamicznego wyboru po nazwie)
PIPELINE_REGISTRY = {
    'kaisti':    kaisti_pipeline,
    'advanced':  advanced_filtering_pipeline,
    'robust':    robust_pipeline,
    'minimal':   minimal_pipeline,
    'wavelet':   wavelet_pipeline,
    'subband':   subband_pipeline,
    'corrected': corrected_pipeline,
    'pca':       pca_pipeline,
}

__all__ = [
    'kaisti_pipeline', 'robust_pipeline', 'minimal_pipeline', 'wavelet_pipeline',
    'subband_pipeline', 'corrected_pipeline', 'pca_pipeline',
    'advanced_filtering_pipeline',
    'ALTERNATIVE_PIPELINES', 'PIPELINE_REGISTRY',
    'aggregate_and_balance_datasets', 'aggregate_balanced_sources',
]

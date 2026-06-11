"""
pipelines.orchestrate — Funkcje agregujące wiele rekordów do tensorów treningowych.

Łączą ładowanie okien z wielu DataFrame'ów w jedną tablicę numpy gotową do treningu.
"""
import numpy as np

from dataset.preprocessor.utils import extract_windows
from .signal import kaisti_pipeline


def aggregate_and_balance_datasets(
    dfs,
    fs=256,
    pipeline_func=kaisti_pipeline,
    seq_len=250,
    stride=None,
    balance=True,
    **pipeline_kwargs,
):
    """
    Przetwarza listę rekordów (DataFrames) i zwraca słownik z tablicami okien.

    Parametry
    ---------
    dfs : list[DataFrame]
        Lista rekordów do przetworzenia.
    pipeline_func : callable
        Funkcja pipeline'u sygnałowego (np. kaisti_pipeline).
    stride : int lub None
        Krok okna. None → stride=seq_len (brak nakładania).
    balance : bool
        True — przycięcie wszystkich rekordów do min_n okien (równa reprezentacja).
        False — wszystkie okna bez przycinania.

    Zwraca
    ------
    dict {'gcg_final', 'scg_final', 'ecg_final'} lub None.
    """
    keys = ['gcg_final', 'scg_final', 'ecg_final']
    all_records_windows = []

    for df in dfs:
        res = pipeline_func(df, fs=fs, **pipeline_kwargs)
        if res is None:
            continue

        gcg_sig = res.get('gcg_final')
        scg_sig = res.get('scg_final')
        ecg_sig = res.get('ecg_final')

        if scg_sig is None or ecg_sig is None:
            continue
        if gcg_sig is None:
            gcg_sig = np.zeros_like(scg_sig)

        windows = extract_windows(
            [gcg_sig, scg_sig, ecg_sig],
            fs,
            seq_len=seq_len,
            clean_mask=res.get('clean_mask'),
            epoch_sec=res.get('epoch_sec', 10),
            stride=stride,
        )
        if len(windows[0]) > 0:
            all_records_windows.append(windows)

    if not all_records_windows:
        print("[Aggregate] Błąd: brak okien.")
        return None

    if balance:
        min_n = min(len(w[0]) for w in all_records_windows)
        print(f"[Aggregate] Balansowanie: {len(all_records_windows)} rekordów × {min_n} okien.")
        aggregated = {k: np.concatenate([w[i][:min_n] for w in all_records_windows])
                      for i, k in enumerate(keys)}
    else:
        total = sum(len(w[0]) for w in all_records_windows)
        print(f"[Aggregate] Bez balansowania: {len(all_records_windows)} rekordów, {total} okien łącznie.")
        aggregated = {k: np.concatenate([w[i] for w in all_records_windows])
                      for i, k in enumerate(keys)}

    return aggregated


def aggregate_balanced_sources(
    dfs_per_source,
    fs=256,
    pipeline_func=kaisti_pipeline,
    seq_len=250,
    stride=None,
    shuffle=True,
    **pipeline_kwargs,
):
    """
    Przetwarza każde źródło osobno, przycina do równej liczby okien i łączy.

    Gwarantuje równą reprezentację zbiorów IEEE i Zenodo w danych treningowych.

    Parametry
    ---------
    dfs_per_source : dict {str: list[DataFrame]}
        Słownik {nazwa_źródła: lista rekordów}.
    shuffle : bool
        Czy tasować okna po konkatenacji (zalecane dla treningu).

    Zwraca
    ------
    dict {'gcg_final', 'scg_final', 'ecg_final'} lub None.
    """
    keys = ['gcg_final', 'scg_final', 'ecg_final']
    source_arrays = {}

    for ds_name, dfs in dfs_per_source.items():
        if not dfs:
            print(f"[Balance źródeł] Źródło '{ds_name}' puste — pomijam.")
            continue
        print(f"\n[Balance źródeł] Przetwarzanie '{ds_name}' ({len(dfs)} rekordów)...")
        data = aggregate_and_balance_datasets(
            dfs,
            fs=fs,
            pipeline_func=pipeline_func,
            seq_len=seq_len,
            stride=stride,
            balance=True,
            **pipeline_kwargs,
        )
        if data is not None and data.get('scg_final') is not None:
            n = len(data['scg_final'])
            source_arrays[ds_name] = data
            print(f"  [{ds_name}] → {n} okien po balansowaniu rekordów.")
        else:
            print(f"  [{ds_name}] → brak okien (wszystkie rekordy odrzucone).")

    if not source_arrays:
        print("[Balance źródeł] Błąd: żadne źródło nie dostarczyło okien.")
        return None

    # Przycięcie każdego źródła do minimalnej liczby okien (równa reprezentacja)
    min_windows = min(len(d['scg_final']) for d in source_arrays.values())
    total = min_windows * len(source_arrays)
    print(f"\n[Balance źródeł] Przycinanie do {min_windows} okien × {len(source_arrays)} źródeł = {total} okien.")
    for ds_name in source_arrays:
        print(f"  {ds_name}: {len(source_arrays[ds_name]['scg_final'])} → {min_windows}")

    combined = {k: [] for k in keys}
    for data in source_arrays.values():
        for k in keys:
            arr = data.get(k)
            combined[k].append(arr[:min_windows] if arr is not None else np.zeros((min_windows, seq_len)))

    result = {k: np.concatenate(combined[k]) for k in keys}

    if shuffle:
        idx = np.random.permutation(len(result['scg_final']))
        result = {k: result[k][idx] for k in keys}

    return result

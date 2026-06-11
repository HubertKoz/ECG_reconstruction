# Rekonstrukcja EKG z sygnałów SCG/GCG

**AGH — Informatyka medyczna, 2026**  
Autor: Hubert Kozierkiewicz | Prowadzący: Dr inż. Szymon Sieciński

Projekt realizuje bezelektrodową rekonstrukcję sygnału EKG z sygnałów sejsmokardiograficznych (SCG) i żyrokardiograficznych (GCG). Porównano pięć konfiguracji model–potok; ewaluacja obejmuje indeksy zmienności rytmu serca (HRV).

---

## Zbiory danych

| Zbiór         | Populacja             | f_s [Hz]  | Rekordów |
|---------------|-----------------------|-----------|----------|
| IEEE DataPort | Zdrowi ochotnicy      | 800 → 256 | 29       |
| Zenodo VHD    | Pacjenci z VHD        | 256       | >100     |

Podział treningowy/walidacyjny: 80/20 na poziomie rekordów (seed=42).  
Rekordy ewaluacyjne: `CP-27, CP-50, UP-28` (Zenodo) oraz `sub_19, sub_22, sub_26` (IEEE).

---

## Konfiguracje model–potok

| Model                  | Pipeline | Param.   | mean r ± σ    | HR MAE [bpm] |
|------------------------|----------|----------|---------------|--------------|
| BiLSTM+Transformer v2  | kaisti   | 2,05 mln | 0,372 ± 0,222 | 9,3          |
| BiLSTM+Transformer v1  | kaisti   | 2,05 mln | 0,363 ± 0,172 | 8,6          |
| TCN                    | wavelet  | 284 tys. | 0,322 ± 0,241 | 12,4         |
| BiLSTM+Transformer PCA | pca      | 2,05 mln | 0,299 ± 0,323 | 10,6         |
| UNet1D                 | subband  | 840 tys. | 0,261 ± 0,219 | 10,5         |

Nazwy modeli do CLI: `bilstm_transformer`, `bilstm_transformer_v2`, `bilstm_transformer_pca`, `tcn`, `unet1d`

---

## Instalacja

```bash
git clone https://github.com/HubertKoz/ECG_reconstruction.git
cd ECG_reconstruction
python -m venv venv && venv\Scripts\activate
# dobierz wersję PyTorch do systemu/CUDA: pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy scikit-learn matplotlib wfdb h5py
```

---

## Użycie

### main.py — trening i ewaluacja

```bash
python main.py                                           # trening + eval, wszystkie modele
python main.py --models bilstm_transformer_v2            # konkretny model
python main.py --eval_only                               # tylko ewaluacja (istniejące wagi)
python main.py --new --epochs 100                        # trening od zera
python main.py --eval_only --filter all                  # ewaluacja wszystkich checkpointów
python main.py --eval_only --records CP-27 sub_19 sub_26
python main.py --compare --compare_epochs 25             # siatka pipeline × architektura
python main.py --info                                    # info o dostępnych danych
```

**Opcje treningu:**

| Flaga         | Domyślnie | Opis |
|---------------|-----------|------|
| `--models`    | wszystkie | Modele do uruchomienia (spacja jako separator) |
| `--epochs`    | `50`      | Liczba epok |
| `--batch_size`| `32`      | Rozmiar batcha |
| `--new`       | false     | Trening od zera (bez tej flagi — wznawia) |

**Opcje ewaluacji:**

| Flaga          | Domyślnie      | Opis |
|----------------|----------------|------|
| `--filter`     | `best`         | Checkpointy do ewaluacji: `best`, `final`, `checkpoint`, `all` |
| `--records`    | z split_info   | Nadpisz rekordy testowe |
| `--n_per_source`| `3`           | Maks. rekordów walidacyjnych per źródło |

### evaluate.py — ewaluacja standalone

```bash
python evaluate.py
python evaluate.py --models bilstm_transformer --filter best final
python evaluate.py --records CP-27 CP-50 UP-28 sub_19 sub_22 sub_26
```

---

## Struktura modułów

```
config.py               # globalne stałe konfiguracyjne
main.py                 # główny punkt wejścia (trening + ewaluacja)
evaluate.py             # ewaluacja checkpointów
utils_peaks.py          # detekcja R-peaków
dataset/
  loader.py             # ładowanie danych IEEE, Zenodo, PhysioNet
  preprocessor/         # filtry, normalizacja, detekcja artefaktów
pipelines/
  signal.py             # potoki preprocessingu (kaisti, pca, wavelet, subband)
  orchestrate.py        # orkiestrator: zbiór danych → okna treningowe
models/
  model.py              # BiLSTM+Transformer (v1 i v2)
  architectures.py      # TCN, UNet1D
  train_global.py       # pętla treningowa cross-subject
evaluation/
  metrics.py            # HRV, wykresy jakości rekonstrukcji
  pipelines.py          # pełna ewaluacja rekordu (predykcja + metryki)
experiments/
  compare_all.py        # siatka eksperymentów pipeline × architektura
data/
  README.md             # szczegółowy opis zbiorów danych i formatów plików
```

---

## Wyniki per rekord

| Rekord | Zbiór  | v1 (r) | v2 (r)    | PCA (r)   | TCN (r)   | UNet1D (r) | Najlepszy |
|--------|--------|--------|-----------|-----------|-----------|------------|-----------|
| CP-27  | Zenodo | 0,501  | 0,106     | **0,838** | 0,315     | 0,113      | PCA       |
| CP-50  | Zenodo | **0,287** | 0,184  | 0,126     | 0,096     | 0,175      | v1        |
| UP-28  | Zenodo | 0,279  | **0,398** | 0,197     | 0,334     | 0,108      | v2        |
| sub_19 | IEEE   | 0,482  | **0,604** | 0,217     | 0,502     | 0,522      | v2        |
| sub_22 | IEEE   | 0,094  | **0,289** | −0,083    | 0,020     | 0,050      | v2        |
| sub_26 | IEEE   | 0,535  | 0,651     | 0,498     | **0,663** | 0,359      | TCN       |

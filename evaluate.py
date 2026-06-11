"""
evaluate.py — Ujednolicona ewaluacja modeli rekonstrukcji EKG.

Zastępuje: evaluate_checkpoints.py, evaluate_val_records.py

Użycie:
  python evaluate.py                               # wszystkie modele, tylko *_best.pth
  python evaluate.py --models bilstm_transformer   # konkretny model
  python evaluate.py --filter all                  # wszystkie checkpointy (wolniej)
  python evaluate.py --filter best final           # tylko best i final
  python evaluate.py --records CP-26 sub_19        # nadpisz rekordy testowe

Struktura wyników:
  results/
    {model}_{pipeline}/
      {record}/
        reconstruction_quality.png
        poincare_gt.png  poincare_pred.png
        hrv_spectrum_gt.png  hrv_spectrum_pred.png
    summary/
      {model}_{pipeline}.json     ← agregat per model
      evaluation_summary.csv      ← wszystkie wiersze
      evaluation_summary.md       ← tabela czytelna dla człowieka
"""

import os
import re
import sys
import csv
import json
import random
import argparse
import numpy as np

# ── Ścieżka projektu ──────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import SPLIT_INFO_PATH, SPLIT_SEED, VAL_RATIO
from evaluation.pipelines import evaluate_full_pipeline

# ── Wzorce nazw plików .pth ───────────────────────────────────────────────────
_CP_PAT  = re.compile(r"^([a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*)_([a-zA-Z0-9]+)_checkpoint_(\d+)\.pth$")
_SP_PAT  = re.compile(r"^([a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*)_([a-zA-Z0-9]+)_(best|final)\.pth$")

# Znane architektury — do odróżnienia nazwy modelu od nazwy pipeline'u
# (parser greedy: bierze najdłuższy prefiks pasujący do architektury)
_KNOWN_MODELS = {
    'bilstm_transformer_v2', 'bilstm_transformer',
    'tcn', 'unet1d',
}


def _parse_pth_name(filename: str):
    """
    Parsuje nazwę pliku .pth → (model_name, pipeline_name, label) lub None.

    Strategie:
    1. Regex dla checkpoint (…_checkpoint_N.pth)
    2. Regex dla best/final (…_best.pth / …_final.pth)
    3. Fallback: wyciągnięcie model_name z _KNOWN_MODELS (greedy prefix)
    """
    stem = filename[:-4]  # bez .pth

    m = _CP_PAT.match(filename)
    if m:
        return m.group(1), m.group(2), f"epoch_{m.group(3)}"

    m = _SP_PAT.match(filename)
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Wycofanie (fallback): sprawdzenie, czy stem zaczyna się od któregoś z _KNOWN_MODELS
    for model in sorted(_KNOWN_MODELS, key=len, reverse=True):
        if stem.startswith(model + '_'):
            rest = stem[len(model) + 1:]
            # rest = pipeline_name(_label)?
            parts = rest.rsplit('_', 1)
            if len(parts) == 2 and parts[1] in ('best', 'final'):
                return model, parts[0], parts[1]
            return model, rest, 'unknown'

    return None


def _sort_label(label: str) -> int:
    if label == 'best':   return 999998
    if label == 'final':  return 999999
    if label == 'unknown': return 0
    m = re.match(r'epoch_(\d+)', label)
    return int(m.group(1)) if m else 0


# ── Pobieranie rekordów walidacyjnych ─────────────────────────────────────────

def get_val_records(n_per_source: int = 3):
    """
    Zwraca {źródło: [rekordy]} na podstawie split_info.json lub dynamicznego podziału.
    """
    if os.path.exists(SPLIT_INFO_PATH):
        try:
            with open(SPLIT_INFO_PATH, 'r', encoding='utf-8') as f:
                split_info = json.load(f)
            val_records = {}
            for ds in ('ieee', 'zenodo'):
                recs = split_info.get(ds, {}).get('val', [])
                if recs:
                    val_records[ds] = recs[:n_per_source]
                    print(f"  [{ds}] split_info.json → {recs[:n_per_source]}")
            if val_records:
                return val_records
            print(f"  [WARN] split_info.json bez rekordów ieee/zenodo — fallback.")
        except Exception as e:
            print(f"  [WARN] Błąd odczytu split_info.json: {e} — fallback.")

    print(f"  [INFO] Odtwarzam podział dynamicznie (seed={SPLIT_SEED}).")
    from dataset import DataLoader
    loader = DataLoader()

    val_records = {}
    rng = random.Random(SPLIT_SEED)

    for ds_name, list_fn in [('ieee', loader.list_ieee), ('zenodo', loader.list_zenodo)]:
        try:
            records = list_fn()
        except Exception:
            continue
        if not records:
            continue
        shuffled = records[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(VAL_RATIO * len(shuffled)))
        val_records[ds_name] = shuffled[:min(n_per_source, n_val)]
        print(f"  [{ds_name}] dynamicznie → {val_records[ds_name]}")

    return val_records


# ── Główna funkcja ewaluacji ──────────────────────────────────────────────────

def run_evaluation(
    models_dir:    str   = 'models',
    results_dir:   str   = 'results',
    filter_labels: list  = None,   # None = ['best'], ['all'] = wszystko
    model_filter:  list  = None,   # None = wszystkie
    records:       list  = None,   # None = z split_info / dynamicznie
    n_per_source:  int   = 3,
):
    """
    Ewaluuje wszystkie dopasowane checkpointy i zapisuje wyniki w results/.

    Parametry
    ---------
    filter_labels : lista dozwolonych etykiet (np. ['best'], ['best', 'final'],
                    lub None/'all' żeby uwzględnić wszystkie checkpointy)
    model_filter  : lista nazw modeli do ewaluacji (None = wszystkie)
    records       : lista rekordów do ewaluacji (None = walidacyjne ze splitu)
    """
    os.makedirs(results_dir, exist_ok=True)
    summary_dir = os.path.join(results_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    # --- Rekordy testowe -------------------------------------------------
    if records:
        # Podział ręczny: rekordy Zenodo zaczynają się od liter CP/UP,
        # IEEE od sub_, PhysioNet od b/m/e
        val_by_source = {}
        for rec in records:
            if rec.startswith('sub_'):
                val_by_source.setdefault('ieee', []).append(rec)
            elif rec.startswith(('b', 'm', 'e')) and rec[1:].isdigit():
                val_by_source.setdefault('physionet', []).append(rec)
            else:
                val_by_source.setdefault('zenodo', []).append(rec)
        print(f"Rekordy nadpisane przez CLI: {val_by_source}")
    else:
        print("Wyznaczanie rekordów walidacyjnych...")
        val_by_source = get_val_records(n_per_source=n_per_source)

    all_records = [(rec, ds) for ds, recs in val_by_source.items() for rec in recs]
    if not all_records:
        print("[BŁĄD] Brak rekordów do ewaluacji.")
        return

    print(f"\nRekordy: {[r for r, _ in all_records]}\n")

    # --- Wczytanie plików .pth -------------------------------------------
    if not os.path.isdir(models_dir):
        print(f"[BŁĄD] Folder '{models_dir}' nie istnieje.")
        return

    pth_files = sorted(f for f in os.listdir(models_dir) if f.endswith('.pth'))
    print(f"Znaleziono {len(pth_files)} plików .pth.")

    # Parsowanie i filtrowanie
    checkpoints = []
    for fname in pth_files:
        parsed = _parse_pth_name(fname)
        if parsed is None:
            print(f"  [SKIP] Nierozpoznany format: {fname}")
            continue
        model_name, pipeline_name, label = parsed

        if model_filter and model_name not in model_filter:
            continue

        if filter_labels is not None and filter_labels != ['all']:
            if not any(
                (fl == label) or
                (fl == 'checkpoint' and label.startswith('epoch_'))
                for fl in filter_labels
            ):
                continue

        checkpoints.append((fname, model_name, pipeline_name, label))

    # Sortowanie: model → pipeline → label (epoki numerycznie)
    checkpoints.sort(key=lambda x: (x[1], x[2], _sort_label(x[3])))

    if not checkpoints:
        print("[BŁĄD] Brak pasujących checkpointów po filtrowaniu.")
        return

    print(f"Ewaluuję {len(checkpoints)} checkpointów × {len(all_records)} rekordów "
          f"= {len(checkpoints) * len(all_records)} przebiegów.\n")

    # --- Pętla ewaluacji -------------------------------------------------
    rows = []

    for (fname, model_name, pipeline_name, label) in checkpoints:
        model_path = os.path.join(models_dir, fname)
        print(f"\n{'='*72}")
        print(f"  {model_name} | {pipeline_name} | {label}")
        print(f"  {fname}")
        print(f"{'='*72}")

        record_results = []

        for record, dataset in all_records:
            try:
                res = evaluate_full_pipeline(
                    model_ecg_path=model_path,
                    record=record,
                    dataset=dataset,
                    model_name=model_name,
                    pipeline_name=pipeline_name,
                )
            except Exception as e:
                print(f"  [WARN] {record}: {e}")
                res = None

            if res is None:
                continue

            record_results.append(res)
            print(f"  {record}: r={res['mean_corr']:.4f}  okien={res['n_windows']}")

            # Wiersz do CSV
            row = {
                'file':          fname,
                'model':         model_name,
                'pipeline':      pipeline_name,
                'label':         label,
                'record':        record,
                'dataset':       dataset,
                'mean_corr':     res['mean_corr'],
                'n_windows':     res['n_windows'],
            }
            for metric in ('HeartRate', 'MeanRR', 'SDNN', 'RMSSD', 'LF_HF_ratio', 'SD1_SD2_ratio'):
                gt   = (res['hrv_gt']   or {}).get(metric)
                pred = (res['hrv_pred'] or {}).get(metric)
                if gt is not None and pred is not None and not (
                    isinstance(gt, float) and np.isnan(gt)
                ) and not (
                    isinstance(pred, float) and np.isnan(pred)
                ):
                    row[f'{metric}_diff'] = abs(float(gt) - float(pred))
                else:
                    row[f'{metric}_diff'] = None
            rows.append(row)

        # Agregat per checkpoint
        if record_results:
            corrs = [r['mean_corr'] for r in record_results if r]
            print(f"\n  ► Śr. r wszystkich rekordów: {np.mean(corrs):.4f} ± {np.std(corrs):.4f}")

    # --- Zapis CSV -------------------------------------------------------
    if not rows:
        print("\n[WARN] Brak wyników do zapisania.")
        return

    csv_path = os.path.join(summary_dir, 'evaluation_summary.csv')
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] CSV: {csv_path}")

    # --- Zapis JSON per model×pipeline -----------------------------------
    from itertools import groupby
    rows_sorted = sorted(rows, key=lambda r: (r['model'], r['pipeline'], _sort_label(r['label'])))
    for (model, pipeline), group in groupby(rows_sorted, key=lambda r: (r['model'], r['pipeline'])):
        group_rows = list(group)
        # Pogrupuj po label → rekordy
        by_label = {}
        for r in group_rows:
            by_label.setdefault(r['label'], []).append(r)

        summary = {
            'model':    model,
            'pipeline': pipeline,
            'labels':   {},
        }
        for lbl, lbl_rows in by_label.items():
            corrs = [r['mean_corr'] for r in lbl_rows if r['mean_corr'] is not None]
            summary['labels'][lbl] = {
                'mean_corr': float(np.mean(corrs)) if corrs else None,
                'records':   {r['record']: r['mean_corr'] for r in lbl_rows},
            }

        json_path = os.path.join(summary_dir, f'{model}_{pipeline}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- Zapis Markdown --------------------------------------------------
    md_path = os.path.join(summary_dir, 'evaluation_summary.md')
    _write_markdown(rows, md_path)
    print(f"[OK] Markdown: {md_path}")
    print(f"\nWszystkie wyniki w: {os.path.abspath(results_dir)}/")


def _write_markdown(rows: list, path: str):
    """Generuje czytelną tabelę MD pogrupowaną model × pipeline."""
    from itertools import groupby

    lines = [
        "# Wyniki Ewaluacji Modeli Rekonstrukcji EKG",
        "",
        f"Rekordy: {sorted(set(r['record'] for r in rows))}",
        "",
    ]

    key_metrics = ['HeartRate', 'SDNN', 'RMSSD']

    sorted_rows = sorted(rows, key=lambda r: (r['model'], r['pipeline'], _sort_label(r['label']), r['record']))

    for (model, pipeline), group in groupby(sorted_rows, key=lambda r: (r['model'], r['pipeline'])):
        lines.append(f"## {model}  ·  pipeline: {pipeline}")
        lines.append("")
        header = "| Label | Rekord | Zbiór | r | n_okien |"
        for m in key_metrics:
            header += f" {m}_diff |"
        lines.append(header)
        sep = "| :--- | :--- | :--- | :---: | :---: |"
        for _ in key_metrics:
            sep += " :---: |"
        lines.append(sep)

        for r in group:
            corr_str = f"{r['mean_corr']:.4f}" if r['mean_corr'] is not None else "N/A"
            row_str = f"| {r['label']} | {r['record']} | {r['dataset']} | {corr_str} | {r['n_windows']} |"
            for m in key_metrics:
                val = r.get(f'{m}_diff')
                row_str += f" {val:.2f} |" if val is not None else " N/A |"
            lines.append(row_str)

        lines.append("")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Ewaluacja modeli rekonstrukcji EKG")
    p.add_argument(
        '--models', nargs='+', default=None,
        help='Filtruj po nazwie modelu (np. bilstm_transformer tcn)'
    )
    p.add_argument(
        '--filter', nargs='+', default=['best'],
        metavar='LABEL',
        help=(
            'Które etykiety checkpointów ewaluować. '
            'Możliwe wartości: best final checkpoint all. '
            'Domyślnie: best'
        )
    )
    p.add_argument(
        '--records', nargs='+', default=None,
        help='Nadpisz rekordy testowe (np. CP-26 sub_19)'
    )
    p.add_argument(
        '--n_per_source', type=int, default=3,
        help='Maks. rekordów na źródło z podziału walidacyjnego (domyślnie 3)'
    )
    p.add_argument(
        '--models_dir', default='models',
        help='Folder z plikami .pth (domyślnie: models/)'
    )
    p.add_argument(
        '--results_dir', default='results',
        help='Folder wynikowy (domyślnie: results/)'
    )
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    filter_labels = args.filter
    if 'all' in filter_labels:
        filter_labels = ['all']

    run_evaluation(
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        filter_labels=filter_labels,
        model_filter=args.models,
        records=args.records,
        n_per_source=args.n_per_source,
    )

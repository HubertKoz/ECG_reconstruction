"""
Porównanie wszystkich kombinacji pipeline'ów preprocessingu i architektur modeli.

Użycie:
  python -m experiments.compare_all
  python -m experiments.compare_all --epochs 30 --data_dir ./data
  python -m experiments.compare_all --pipelines kaisti minimal --models bilstm_transformer tcn
  python -m experiments.compare_all --epochs 5 --synthetic  # szybki test na danych syntetycznych

Wyniki zapisywane są w katalogu results/:
  results/
    comparison_summary.json   – zestawienie wszystkich przebiegów
    comparison_plot.png        – wykres słupkowy korelacji walidacyjnych
    runs/
      {pipeline}__{model}/
        metrics.json           – per-epoka loss i korelacja
        loss_curve.png         – krzywa uczenia
        model_best.pth         – najlepsze wagi
"""

import os
import sys
import json
import random
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

# ── Ścieżka projektu ──────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipelines import kaisti_pipeline, advanced_filtering_pipeline, aggregate_and_balance_datasets
from pipelines import ALTERNATIVE_PIPELINES
from models.architectures import ARCHITECTURE_REGISTRY, count_parameters

# ── Konfiguracja ─────────────────────────────────────────────────────────────

PIPELINES = {
    'kaisti':    kaisti_pipeline,
    'advanced':  advanced_filtering_pipeline,
    **ALTERNATIVE_PIPELINES,          # minimal, wavelet, robust
}

RESULTS_DIR = os.path.join(_ROOT, 'results')
RUNS_DIR    = os.path.join(RESULTS_DIR, 'runs')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LEN  = 250   # próbki (1 s przy 256 Hz)
FS       = 256
SEED     = 42


# ── Dane syntetyczne (fallback) ───────────────────────────────────────────────

def _synthetic_ecg(n_samples, fs=256, hr_bpm=70):
    """Uproszczony syntetyczny sygnał EKG (Gaussowski zespół QRS + szum)."""
    t = np.arange(n_samples) / fs
    rr = 60.0 / hr_bpm  # s
    ecg = np.zeros(n_samples)
    beat_times = np.arange(0, t[-1], rr)
    for bt in beat_times:
        ecg += np.exp(-((t - bt) ** 2) / (2 * (0.015 ** 2)))  # R-peak
        ecg += 0.15 * np.exp(-((t - bt - 0.12) ** 2) / (2 * (0.04 ** 2)))  # T-wave
        ecg += 0.08 * np.exp(-((t - bt - 0.16) ** 2) / (2 * (0.025 ** 2)))  # P-wave (pre-next)
    return (ecg + np.random.randn(n_samples) * 0.05)


def _synthetic_scg(n_samples, fs=256, hr_bpm=70, phase_shift=0.05):
    """SCG: mechaniczne opóźnienie ~50 ms + inny kształt."""
    t = np.arange(n_samples) / fs
    rr = 60.0 / hr_bpm
    scg = np.zeros(n_samples)
    for bt in np.arange(0, t[-1], rr):
        bt_s = bt + phase_shift
        scg += np.exp(-((t - bt_s) ** 2) / (2 * (0.02 ** 2)))
        scg -= 0.3 * np.exp(-((t - bt_s - 0.04) ** 2) / (2 * (0.015 ** 2)))
    return (scg + np.random.randn(n_samples) * 0.08)


def make_synthetic_dataset(n_windows=400, seq_len=SEQ_LEN, fs=FS):
    """Generuje syntetyczny zbiór okien (pcg, scg, ecg)."""
    total = n_windows * seq_len + 1000
    ecg_sig = _synthetic_ecg(total, fs)
    scg_sig = _synthetic_scg(total, fs, phase_shift=0.05)
    gcg_sig = _synthetic_scg(total, fs, phase_shift=0.08)  # GCG jako pcg

    def _zscore(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    ecg_sig = _zscore(ecg_sig)
    scg_sig = _zscore(scg_sig)
    gcg_sig = _zscore(gcg_sig)

    pcgs, scgs, ecgs = [], [], []
    for i in range(n_windows):
        s, e = i * seq_len, i * seq_len + seq_len
        pcgs.append(gcg_sig[s:e])
        scgs.append(scg_sig[s:e])
        ecgs.append(ecg_sig[s:e])

    return (np.array(pcgs), np.array(scgs), np.array(ecgs))


# ── Przygotowanie danych z prawdziwego pipeline'u ────────────────────────────

def load_real_data(data_dir, pipeline_name, pipeline_fn, seq_len=SEQ_LEN):
    """
    Wczytuje dane przez wskazany pipeline.
    Zwraca (pcg_arr, scg_arr, ecg_arr) lub None jeśli brak danych.
    """
    try:
        from dataset import DataLoader as ECGDataLoader
        loader = ECGDataLoader(base_data_dir=data_dir)

        all_dfs = []
        for ds_name, dfs in loader.load_all_datasets().items():
            all_dfs.extend(dfs)

        if not all_dfs:
            return None

        balanced = aggregate_and_balance_datasets(
            all_dfs, fs=FS, pipeline_func=pipeline_fn, seq_len=seq_len
        )
        if balanced is None or balanced['scg_final'] is None or balanced['ecg_final'] is None:
            return None

        return balanced['gcg_final'], balanced['scg_final'], balanced['ecg_final']

    except Exception as e:
        print(f"  [WARN] Nie udało się załadować danych pipeline '{pipeline_name}': {e}")
        return None


def make_torch_loaders(pcg_arr, scg_arr, ecg_arr, train_frac=0.8, batch_size=32):
    """Dzieli dane na train/val i tworzy DataLoadery."""
    n = len(pcg_arr)
    idx = list(range(n))
    random.shuffle(idx)
    n_train = int(train_frac * n)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    def _tensor(arr, idxs):
        return torch.tensor(arr[idxs], dtype=torch.float32).unsqueeze(-1)

    tr_ds = TensorDataset(_tensor(pcg_arr, tr_idx), _tensor(scg_arr, tr_idx), _tensor(ecg_arr, tr_idx))
    va_ds = TensorDataset(_tensor(pcg_arr, va_idx), _tensor(scg_arr, va_idx), _tensor(ecg_arr, va_idx))

    tr_ld = TorchDataLoader(tr_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    va_ld = TorchDataLoader(va_ds, batch_size=batch_size, shuffle=False)
    return tr_ld, va_ld


# ── Pętla treningowa ─────────────────────────────────────────────────────────

def _pearson_batch(pred, target):
    """Śr. korelacja Pearsona po batchu (po wymiarze czasowym)."""
    p = pred.squeeze(-1);   t = target.squeeze(-1)   # [B, T]
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = (p.norm(dim=1) * t.norm(dim=1)).clamp(min=1e-8)
    return (num / den).mean().item()


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for pcg, scg, ecg in loader:
        pcg, scg, ecg = pcg.to(device), scg.to(device), ecg.to(device)
        optimizer.zero_grad()
        out = model(pcg, scg)
        loss = criterion(out, ecg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_corr = 0.0
    for pcg, scg, ecg in loader:
        pcg, scg, ecg = pcg.to(device), scg.to(device), ecg.to(device)
        out = model(pcg, scg)
        total_loss += criterion(out, ecg).item()
        total_corr += _pearson_batch(out, ecg)
    return total_loss / len(loader), total_corr / len(loader)


# ── Zapis wyników ─────────────────────────────────────────────────────────────

def save_loss_curve(run_dir, history, pipeline_name, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], label='Train MSE')
    ax1.plot(epochs, history['val_loss'],   label='Val MSE')
    ax1.set_title(f'Strata MSE\n{pipeline_name} + {model_name}')
    ax1.set_xlabel('Epoka'); ax1.set_ylabel('MSE'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['val_corr'], color='green', label='Val Pearson r')
    ax2.axhline(max(history['val_corr']), color='red', linestyle='--',
                label=f"max r = {max(history['val_corr']):.4f}")
    ax2.set_title(f'Korelacja Pearsona (val)\n{pipeline_name} + {model_name}')
    ax2.set_xlabel('Epoka'); ax2.set_ylabel('Pearson r'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(run_dir, 'loss_curve.png')
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def save_comparison_plot(summary, out_dir):
    """
    Wykres słupkowy: os X = kombinacja pipeline+model, os Y = najlepsza korelacja val.
    Dwie warstwy: kolor = pipeline, wzór = architektura.
    """
    if not summary:
        return

    labels   = [f"{r['pipeline']}\n{r['model']}" for r in summary]
    best_corr = [r['best_val_corr'] for r in summary]
    pipelines  = sorted(set(r['pipeline'] for r in summary))
    colors     = plt.cm.tab10(np.linspace(0, 0.8, len(pipelines)))
    cmap       = {p: c for p, c in zip(pipelines, colors)}

    fig, ax = plt.subplots(figsize=(max(10, len(summary) * 1.4), 6))
    bar_colors = [cmap[r['pipeline']] for r in summary]
    bars = ax.bar(range(len(summary)), best_corr, color=bar_colors, edgecolor='black', linewidth=0.5)

    # Wartości nad słupkami
    for bar, val in zip(bars, best_corr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Najlepsza korelacja Pearsona (val)')
    ax.set_title('Porównanie: pipeline preprocessingu × architektura modelu')
    ax.set_ylim(0, min(1.05, max(best_corr) + 0.1))
    ax.grid(True, axis='y', alpha=0.3)

    # Legenda dla pipeline'ów (kolor)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cmap[p], label=p) for p in pipelines],
              title='Pipeline', loc='lower right')

    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison_plot.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n[Comparison] Wykres zapisany: {path}")
    return path


# ── Główna pętla porównań ─────────────────────────────────────────────────────

def run_comparison(
    pipeline_names, model_names,
    data_dir='./data', epochs=25, batch_size=32,
    use_synthetic=False, n_synthetic=600
):
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    os.makedirs(RUNS_DIR, exist_ok=True)

    summary_path = os.path.join(RESULTS_DIR, 'comparison_summary.json')
    # Wczytanie istniejących wyników (wznowienie po awarii/crashu)
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        done_keys = {(r['pipeline'], r['model']) for r in summary}
    else:
        summary = []
        done_keys = set()

    criterion = nn.MSELoss()

    for pipe_name in pipeline_names:
        pipe_fn = PIPELINES[pipe_name]

        print(f"\n{'='*60}")
        print(f"  PIPELINE: {pipe_name}")
        print(f"{'='*60}")

        # Wczytanie danych dla tego pipeline'u
        if use_synthetic:
            print("  Generowanie danych syntetycznych...")
            pcg_arr, scg_arr, ecg_arr = make_synthetic_dataset(n_windows=n_synthetic)
            data_source = 'synthetic'
        else:
            print(f"  Wczytywanie danych z {data_dir} przez pipeline '{pipe_name}'...")
            result = load_real_data(data_dir, pipe_name, pipe_fn)
            if result is None:
                print(f"  -> Brak danych — fallback na syntetyczne.")
                pcg_arr, scg_arr, ecg_arr = make_synthetic_dataset(n_windows=n_synthetic)
                data_source = 'synthetic'
            else:
                pcg_arr, scg_arr, ecg_arr = result
                data_source = 'real'

        print(f"  Dane: {len(pcg_arr)} okien ({data_source})")
        tr_ld, va_ld = make_torch_loaders(pcg_arr, scg_arr, ecg_arr, batch_size=batch_size)

        for model_name in model_names:
            run_key = (pipe_name, model_name)
            if run_key in done_keys:
                print(f"\n  [SKIP] {pipe_name} + {model_name} — już ukończony.")
                continue

            run_id  = f"{pipe_name}__{model_name}"
            run_dir = os.path.join(RUNS_DIR, run_id)
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n  -- Model: {model_name} --")
            ModelClass = ARCHITECTURE_REGISTRY[model_name]
            model = ModelClass().to(device)
            n_params = count_parameters(model)
            print(f"  Parametrów trenowalnych: {n_params:,}")

            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

            history = {'train_loss': [], 'val_loss': [], 'val_corr': []}
            best_corr = -1.0
            best_epoch = 0
            t0 = time.time()

            for epoch in range(1, epochs + 1):
                tr_loss = train_one_epoch(model, tr_ld, optimizer, criterion)
                va_loss, va_corr = validate_one_epoch(model, va_ld, criterion)
                scheduler.step()

                history['train_loss'].append(round(tr_loss, 6))
                history['val_loss'].append(round(va_loss, 6))
                history['val_corr'].append(round(va_corr, 6))

                if va_corr > best_corr:
                    best_corr  = va_corr
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(run_dir, 'model_best.pth'))

                elapsed = time.time() - t0
                print(f"  Ep {epoch:3d}/{epochs}  "
                      f"TrainMSE={tr_loss:.4f}  ValMSE={va_loss:.4f}  "
                      f"ValR={va_corr:.4f}  BestR={best_corr:.4f}  "
                      f"[{elapsed:.0f}s]")

            # Zapis wyników
            run_result = {
                'pipeline':       pipe_name,
                'model':          model_name,
                'n_params':       n_params,
                'n_windows':      int(len(pcg_arr)),
                'data_source':    data_source,
                'epochs':         epochs,
                'best_val_corr':  round(best_corr, 6),
                'best_epoch':     best_epoch,
                'final_val_loss': round(history['val_loss'][-1], 6),
                'history':        history,
            }
            with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
                json.dump(run_result, f, indent=2)

            save_loss_curve(run_dir, history, pipe_name, model_name)
            print(f"  -> Wyniki zapisane w: {run_dir}/")

            # Dołącz do summary i zapisz na bieżąco
            summary_entry = {k: v for k, v in run_result.items() if k != 'history'}
            summary.append(summary_entry)
            done_keys.add(run_key)
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

    # Ranking i wykres zbiorczy
    summary_sorted = sorted(summary, key=lambda r: r['best_val_corr'], reverse=True)
    print(f"\n{'='*60}")
    print("  RANKING (najlepsza korelacja walidacyjna)")
    print(f"{'='*60}")
    print(f"  {'#':<3} {'Pipeline':<12} {'Model':<22} {'BestR':>7} {'BestEp':>7} {'Params':>9}")
    print(f"  {'-' * 65}")
    for i, r in enumerate(summary_sorted, 1):
        print(f"  {i:<3} {r['pipeline']:<12} {r['model']:<22} "
              f"{r['best_val_corr']:>7.4f} {r['best_epoch']:>7d} {r['n_params']:>9,}")

    with open(summary_path, 'w') as f:
        json.dump(summary_sorted, f, indent=2)

    save_comparison_plot(summary_sorted, RESULTS_DIR)
    print(f"\n[DONE] Pełne wyniki w: {RESULTS_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Porównanie pipeline'ów i architektur modeli rekonstrukcji EKG"
    )
    parser.add_argument('--epochs',     type=int,   default=25,
                        help='Liczba epok dla każdego przebiegu (domyślnie 25)')
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--data_dir',   type=str,   default='./data',
                        help='Katalog z danymi (Zenodo/, IEEE/, PhysioNet/)')
    parser.add_argument('--synthetic',  action='store_true',
                        help='Wymuszenie użycia danych syntetycznych (szybki test)')
    parser.add_argument('--n_synthetic', type=int,  default=600,
                        help='Liczba okien syntetycznych gdy --synthetic')
    parser.add_argument('--pipelines',  nargs='+',
                        choices=list(PIPELINES.keys()),
                        default=list(PIPELINES.keys()),
                        help='Które pipeline\'y testować')
    parser.add_argument('--models',     nargs='+',
                        choices=list(ARCHITECTURE_REGISTRY.keys()),
                        default=list(ARCHITECTURE_REGISTRY.keys()),
                        help='Które architektury testować')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Device: {device}")
    print(f"Pipeline'y: {args.pipelines}")
    print(f"Modele:     {args.models}")
    print(f"Epoki:      {args.epochs}")
    print(f"Dane:       {'syntetyczne' if args.synthetic else args.data_dir}")
    run_comparison(
        pipeline_names=args.pipelines,
        model_names=args.models,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_synthetic=args.synthetic,
        n_synthetic=args.n_synthetic,
    )

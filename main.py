"""
main.py — Glowny punkt wejscia projektu.

Trenuje wybrane modele, a nastepnie ewaluuje je na rekordach walidacyjnych.
Zastepuje run_final_experiments.py.

Uzycie:
  python main.py                               # train + eval, wszystkie modele
  python main.py --models bilstm_transformer   # konkretny model
  python main.py --eval_only                   # pomin trening
  python main.py --epochs 200 --new            # trening od zera
  python main.py --filter all                  # eval wszystkich checkpointow
  python main.py --info                        # info o zbiorach danych
  python main.py --compare                     # porownanie pipeline x architektura
"""

import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Predefiniowane pary model-pipeline
PAIRS = {
    'bilstm_transformer':    'kaisti',
    'bilstm_transformer_v2': 'kaisti',
    'tcn':                   'wavelet',
    'unet1d':                'subband',
    'bilstm_transformer_pca': 'pca',
}


def _parse():
    p = argparse.ArgumentParser(
        description="Rekonstrukcja EKG z sygnalow SCG/GCG — trening i ewaluacja",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Tryby specjalne
    p.add_argument(
        '--info', action='store_true',
        help='Wyswietl info o dostepnych zbiorach danych i zakoncz'
    )
    p.add_argument(
        '--compare', action='store_true',
        help='Porownaj kombinacje pipeline x architektura\n(uruchamia experiments/compare_all.py)'
    )

    # Wybor modeli
    p.add_argument(
        '--models', nargs='+', default=list(PAIRS.keys()),
        metavar='MODEL',
        help='Modele do uruchomienia. Domyslnie: wszystkie\nDostepne: ' + str(list(PAIRS.keys()))
    )

    # Trening
    p.add_argument(
        '--eval_only', action='store_true',
        help='Pomin trening — tylko ewaluacja istniejacych wag'
    )
    p.add_argument('--epochs',     type=int, default=50,
                   help='Liczba epok treningu (domyslnie: 50)')
    p.add_argument('--batch_size', type=int, default=32,
                   help='Rozmiar batcha (domyslnie: 32)')
    p.add_argument('--new',        action='store_true',
                   help='Zacznij trening od zera (bez tej flagi wznawia z istniejacych wag)')

    # Ewaluacja
    p.add_argument(
        '--filter', nargs='+', default=['best'],
        metavar='LABEL',
        help='Ktore checkpointy ewaluowac po treningu.\nWartosci: best  final  checkpoint  all\nDomyslnie: best'
    )
    p.add_argument(
        '--records', nargs='+', default=None,
        metavar='RECORD',
        help='Nadpisz rekordy testowe (domyslnie: z data/cache/split_info.json)'
    )
    p.add_argument(
        '--n_per_source', type=int, default=3,
        help='Maks. rekordow walidacyjnych per zrodlo (domyslnie: 3)'
    )

    # Opcje dla --compare
    p.add_argument('--compare_epochs',    type=int,   default=25)
    p.add_argument('--compare_synthetic', action='store_true')
    p.add_argument('--compare_pipelines', nargs='+',  default=None)
    p.add_argument('--compare_models',    nargs='+',  default=None)

    return p.parse_args()


def _header(text):
    print(f"\n{'='*72}\n  {text}\n{'='*72}")


def main():
    args = _parse()

    # --info
    if args.info:
        from dataset import DataLoader
        _header("INFORMACJE O ZBIORACH DANYCH")
        loader = DataLoader()
        all_data = loader.load_all_datasets()
        print()
        for ds_name, dfs in all_data.items():
            total_rows = sum(len(df) for df in dfs)
            print(f"  {ds_name.upper():<12} {len(dfs):>3} rekordow   {total_rows:>10} probek")
        print()
        return

    # --compare
    if args.compare:
        from experiments.compare_all import run_comparison, PIPELINES, ARCHITECTURE_REGISTRY
        _header("POROWNANIE PIPELINE'OW x ARCHITEKTUR")
        run_comparison(
            pipeline_names = args.compare_pipelines or list(PIPELINES.keys()),
            model_names    = args.compare_models    or list(ARCHITECTURE_REGISTRY.keys()),
            epochs         = args.compare_epochs,
            use_synthetic  = args.compare_synthetic,
        )
        return

    # Walidacja modeli
    selected = [m for m in args.models if m in PAIRS]
    unknown  = [m for m in args.models if m not in PAIRS]
    if unknown:
        print(f"[WARN] Nieznane modele (pomijam): {unknown}")
    if not selected:
        print(f"[BLAD] Zaden model nie pasuje do PAIRS: {list(PAIRS.keys())}")
        sys.exit(1)

    filter_labels = ['all'] if 'all' in args.filter else args.filter

    _header("REKONSTRUKCJA EKG — START")
    print(f"  Modele:      {selected}")
    print(f"  Epoki:       {args.epochs}   Batch: {args.batch_size}   Nowy: {args.new}")
    print(f"  Eval only:   {args.eval_only}")
    print(f"  Eval filter: {filter_labels}")
    if args.records:
        print(f"  Rekordy:     {args.records}")

    # 1. Trening
    if not args.eval_only:
        from models.train_global import main as train_model
        for model_name in selected:
            pipeline_name = PAIRS[model_name]
            _header(f"TRENING: {model_name}  ·  pipeline: {pipeline_name}")
            train_model(
                model_name    = model_name,
                pipeline_name = pipeline_name,
                epochs        = args.epochs,
                batch_size    = args.batch_size,
                resume        = not args.new,
            )

    # 2. Ewaluacja
    _header("EWALUACJA")
    from evaluate import run_evaluation
    run_evaluation(
        model_filter  = selected,
        filter_labels = filter_labels,
        records       = args.records,
        n_per_source  = args.n_per_source,
    )


if __name__ == '__main__':
    main()

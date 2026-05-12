import argparse
import sys
import os

from models.train_global import main as train_ecg_model
from evaluation import evaluate_reconstruction_pipeline

def main():
    parser = argparse.ArgumentParser(description="Pipeline rekonstrukcji EKG z sygnałów SCG/GCG")

    parser.add_argument('--train',   action='store_true', help="Trenowanie modelu rekonstrukcji EKG")
    parser.add_argument('--eval',    action='store_true', help="Ewaluacja wytrenowanego modelu")
    parser.add_argument('--all',     action='store_true', help="Trenowanie + ewaluacja")
    parser.add_argument('--info',    action='store_true', help="Informacje o dostępnych zbiorach danych")
    parser.add_argument('--compare', action='store_true',
                        help="Porównanie wszystkich pipeline'ów × architektur (experiments/compare_all.py)")

    parser.add_argument('--model_path',   type=str, default='models/global_best_ecg_model.pth')
    parser.add_argument('--eval_record',  type=str, default='CP-01')
    parser.add_argument('--eval_samples', type=int, default=3)

    # Opcje dla --compare
    parser.add_argument('--compare_epochs',     type=int, default=25,
                        help='Liczba epok na przebieg (dla --compare)')
    parser.add_argument('--compare_synthetic',  action='store_true',
                        help='Użyj danych syntetycznych zamiast prawdziwych (szybki test)')
    parser.add_argument('--compare_pipelines',  nargs='+', default=None,
                        help='Pipeline\'y do porównania (domyślnie wszystkie)')
    parser.add_argument('--compare_models',     nargs='+', default=None,
                        help='Architektury do porównania (domyślnie wszystkie)')
    
    args = parser.parse_args()

    if len(sys.argv) == 1:
        print("Nie podano flag. Domyślnie uruchamiam pełny pipeline (trenowanie + ewaluacja).")
        print("Opcje: --train  --eval  --all  --compare  --info  --help\n")
        args.all = True

    if args.all:
        args.train = True
        args.eval  = True

    if args.compare:
        print("="*60)
        print(" PORÓWNANIE PIPELINE'ÓW × ARCHITEKTUR")
        print("="*60)
        from experiments.compare_all import run_comparison, PIPELINES, ARCHITECTURE_REGISTRY
        pipe_names  = args.compare_pipelines or list(PIPELINES.keys())
        model_names = args.compare_models    or list(ARCHITECTURE_REGISTRY.keys())
        run_comparison(
            pipeline_names=pipe_names,
            model_names=model_names,
            epochs=args.compare_epochs,
            use_synthetic=args.compare_synthetic,
        )
        return

    if args.train:
        print("="*60)
        print(" ROZPOCZYNANIE TRENINGU MODELU REKONSTRUKCJI EKG")
        print("="*60)
        train_ecg_model()
        print("\n[INFO] Trening zakończony.\n")
        
    if args.eval:
        print("="*60)
        print(f" ROZPOCZYNANIE EWALUACJI MODELU REKONSTRUKCJI EKG")
        print(f" Wagi: {args.model_path}, Rekord: {args.eval_record}")
        print("="*60)
        
        if not os.path.exists(args.model_path):
            print(f"[BŁĄD] Nie znaleziono pliku wag modelu: {args.model_path}")
            print("Najpierw musisz wytrenować model, np. używając flagi --train lub --all.")
            sys.exit(1)
            
        # Wywołanie nowej funkcji z pakietu evaluation
        evaluate_reconstruction_pipeline(
            model_path=args.model_path, 
            record=args.eval_record, 
            num_samples=args.eval_samples
        )
        print("\n[INFO] Ewaluacja zakończona.\n")

    if args.info:
        from data_loader import DataLoader
        print("="*60)
        print(" INFORMACJE O ZBIORACH DANYCH")
        print("="*60)
        loader = DataLoader()
        all_data = loader.load_all_datasets()
        
        print("\nPodsumowanie wczytanych danych:")
        for ds_name, dfs in all_data.items():
            total_rows = sum([len(df) for df in dfs])
            print(f" - {ds_name.upper():<10}: {len(dfs):>3} rekordów, łącznie {total_rows:>10} próbek")
        print("="*60)

if __name__ == "__main__":
    main()

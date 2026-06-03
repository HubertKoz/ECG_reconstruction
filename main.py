import argparse
import sys
import os

from models.train_global import main as train_ecg_model
from evaluation import evaluate_reconstruction_pipeline

def main():
    parser = argparse.ArgumentParser(description="Pipeline rekonstrukcji EKG z sygnałów SCG/GCG")

    parser.add_argument('--train',   action='store_true', help="Trenowanie modelu rekonstrukcji EKG")
    parser.add_argument('--eval',    action='store_true', help="Ewaluacja wytrenowanego modelu")
    parser.add_argument('--eval_full', action='store_true', help="Pełna ewaluacja end-to-end (Rekonstrukcja + HRV)")
    parser.add_argument('--all',     action='store_true', help="Trenowanie + ewaluacja")
    parser.add_argument('--info',    action='store_true', help="Informacje o dostępnych zbiorach danych")
    parser.add_argument('--compare', action='store_true',
                        help="Porównanie wszystkich pipeline'ów × architektur (experiments/compare_all.py)")

    parser.add_argument('--model_path',   type=str, default=None, help="Ścieżka do wag modelu (.pth). Domyślnie dopasowuje do wybranego modelu i potoku.")
    parser.add_argument('--eval_record',  type=str, default='CP-03')
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
    
    # Główne parametry wyboru modelu i preprocessingu
    parser.add_argument('--model',      type=str, default='bilstm_transformer', help="Model: unet1d, bilstm_transformer, cnn_bilstm, tcn, resnet1d")
    parser.add_argument('--pipeline',   type=str, default='kaisti',             help="Pipeline: kaisti, subband, wavelet, robust, minimal")
    parser.add_argument('--epochs',     type=int, default=50,                  help="Liczba epok treningu")
    parser.add_argument('--batch_size', type=int, default=32,                  help="Rozmiar batcha")
    parser.add_argument('--new',        action='store_true',                   help="Rozpocznij trening od nowa (domyślnie wznawia z najlepszych wag)")
    parser.add_argument('--preprocess', action='store_true',                   help="Wymuś preprocessing danych i zaktualizuj cache")
    
    args = parser.parse_args()

    # Dynamiczne dopasowanie ścieżki modelu w zależności od wybranych flag, jeśli nie została podana ręcznie
    if args.model_path is None:
        args.model_path = f"models/{args.model}_{args.pipeline}_best.pth"
        # Fallback do starej domyślnej nazwy, jeśli nowa jeszcze nie istnieje
        if not os.path.exists(args.model_path) and os.path.exists("models/global_best_ecg_model.pth"):
            args.model_path = "models/global_best_ecg_model.pth"

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
        print(f" ROZPOCZYNANIE TRENINGU REKONSTRUKCJI EKG ({args.model} + {args.pipeline})")
        print("="*60)
        train_ecg_model(
            model_name=args.model,
            pipeline_name=args.pipeline,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume=(not args.new),
            force_preprocess=args.preprocess
        )
        print("\n[INFO] Trening zakończony.\n")
        
    if args.eval:
        print("="*60)
        print(f" ROZPOCZYNANIE EWALUACJI MODELU REKONSTRUKCJI EKG")
        print(f" Model: {args.model}, Preprocessing: {args.pipeline}")
        print(f" Wagi: {args.model_path}, Rekord: {args.eval_record}")
        print("="*60)
        
        if not os.path.exists(args.model_path):
            print(f"[BŁĄD] Nie znaleziono pliku wag modelu: {args.model_path}")
            print("Najpierw musisz wytrenować model, np. używając flagi --train lub --all.")
            sys.exit(1)
            
        # 1. Ewaluacja na rekordzie głównym
        main_record = args.eval_record
        evaluate_reconstruction_pipeline(
            model_path=args.model_path, 
            record=main_record, 
            num_samples=args.eval_samples,
            base_data_dir='./data',
            model_name=args.model,
            pipeline_name=args.pipeline
        )
        
        # 2. Ewaluacja na rekordzie z alternatywnego zbioru (badanie generalizacji)
        alt_record = "sub_6" if not main_record.startswith("sub_") else "CP-03"
        print("\n" + "="*60)
        print(f" BADANIE GENERALIZACJI: EWALUACJA NA ALTERNATYWNYM ZBIORZE")
        print(f" Rekord: {alt_record}")
        print("="*60)
        evaluate_reconstruction_pipeline(
            model_path=args.model_path, 
            record=alt_record, 
            num_samples=args.eval_samples,
            base_data_dir='./data',
            model_name=args.model,
            pipeline_name=args.pipeline
        )
        print("\n[INFO] Ewaluacja zakończona.\n")

    if args.eval_full:
        print("="*60)
        print(f" ROZPOCZYNANIE PEŁNEJ EWALUACJI END-TO-END (Rekonstrukcja + HRV)")
        print(f" Model: {args.model}, Preprocessing: {args.pipeline}")
        print(f" Wagi: {args.model_path}, Rekord: {args.eval_record}")
        print("="*60)
        from evaluation.pipelines import evaluate_full_pipeline
        
        # 1. Pełna ewaluacja na rekordzie głównym
        main_record = args.eval_record
        evaluate_full_pipeline(
            model_ecg_path=args.model_path,
            record=main_record,
            dataset='Zenodo',
            base_data_dir='./data',
            model_name=args.model,
            pipeline_name=args.pipeline
        )
        
        # 2. Pełna ewaluacja na rekordzie z alternatywnego zbioru (badanie generalizacji)
        alt_record = "sub_6" if not main_record.startswith("sub_") else "CP-03"
        print("\n" + "="*60)
        print(f" BADANIE GENERALIZACJI: EWALUACJA NA ALTERNATYWNYM ZBIORZE")
        print(f" Rekord: {alt_record}")
        print("="*60)
        evaluate_full_pipeline(
            model_ecg_path=args.model_path,
            record=alt_record,
            dataset='Zenodo',
            base_data_dir='./data',
            model_name=args.model,
            pipeline_name=args.pipeline
        )
        print("\n[INFO] Pełna ewaluacja zakończona.\n")

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

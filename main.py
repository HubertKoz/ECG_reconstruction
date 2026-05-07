import argparse
import sys
import os

# Importy z istniejących skryptów
from models.train_global import main as train_ecg_model
from evaluate_model import evaluate_and_plot

def main():
    parser = argparse.ArgumentParser(description="Główny skrypt uruchamiający pipeline rekonstrukcji EKG (Model Ogólny)")
    
    parser.add_argument('--train', action='store_true', help="Uruchamia trenowanie modelu rekonstrukcji EKG")
    parser.add_argument('--eval', action='store_true', help="Uruchamia ewaluację wytrenowanego modelu rekonstrukcji EKG")
    parser.add_argument('--all', action='store_true', help="Uruchamia cały pipeline: trenowanie, a następnie ewaluację")
    
    parser.add_argument('--model_path', type=str, default='models/global_best_ecg_model.pth', help="Ścieżka do pliku z wagami modelu (używane przy ewaluacji)")
    parser.add_argument('--eval_record', type=str, default='CP-01', help="Nazwa rekordu do ewaluacji (np. 'CP-01' z Zenodo lub 'sub_1' z IEEE)")
    parser.add_argument('--eval_samples', type=int, default=3, help="Liczba próbek (okien) do wyświetlenia na wykresach ewaluacyjnych")
    
    args = parser.parse_args()
    
    # Jeśli nie podano żadnych argumentów, domyślnie uruchom pełny pipeline (zgodnie z intencją "uruchomienia całego pipeline")
    if len(sys.argv) == 1:
        print("Nie podano flag trybu. Domyślnie uruchamiam pełny pipeline (trenowanie + ewaluacja).")
        print("Aby zobaczyć wszystkie opcje, użyj: python main.py --help\n")
        args.all = True
        
    if args.all:
        args.train = True
        args.eval = True

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
            
        evaluate_and_plot(model_path=args.model_path, record=args.eval_record, num_samples=args.eval_samples)
        print("\n[INFO] Ewaluacja zakończona.\n")

if __name__ == "__main__":
    main()

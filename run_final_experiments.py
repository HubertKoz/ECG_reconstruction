import os
import sys
import argparse
import numpy as np

# Dodanie ścieżki głównego katalogu
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from models.train_global import main as train_model
from evaluation.pipelines import evaluate_full_pipeline

PAIRS = {
    'bilstm_transformer': 'kaisti',
    'tcn': 'wavelet',
    'unet1d': 'subband'
}

def main():
    parser = argparse.ArgumentParser(description="Zautomatyzowane uruchamianie wybranych eksperymentów EKG")
    parser.add_argument('--models', nargs='+', default=['bilstm_transformer', 'tcn', 'unet1d'],
                        help="Modele do uruchomienia (dostępne: bilstm_transformer, tcn, unet1d)")
    parser.add_argument('--epochs', type=int, default=50, help="Liczba epok treningu")
    parser.add_argument('--batch_size', type=int, default=32, help="Rozmiar batcha")
    parser.add_argument('--new', action='store_true', help="Rozpocznij trening od nowa (zignoruj wagi)")
    parser.add_argument('--preprocess', action='store_true', help="Wymuś preprocessing danych i zaktualizuj cache")
    parser.add_argument('--eval_only', action='store_true', help="Uruchom tylko ewaluację na istniejących modelach")
    args = parser.parse_args()

    selected_models = [m for m in args.models if m in PAIRS]
    if not selected_models:
        print(f"[BŁĄD] Nie wybrano żadnych prawidłowych modeli z zestawu: {list(PAIRS.keys())}")
        sys.exit(1)

    print("="*80)
    print(" ROZPOCZYNANIE ZAUTOMATYZOWANYCH EKSPERYMENTÓW PORÓWNAWCZYCH")
    print(f" Wybrane modele: {selected_models}")
    print(f" Epoki: {args.epochs} | Batch size: {args.batch_size}")
    print(f" Nowy trening: {args.new} | Wymuszenie preprocessingu: {args.preprocess}")
    print(f" Tryb tylko ewaluacja: {args.eval_only}")
    print("="*80)

    results_report = {}

    for model in selected_models:
        pipeline = PAIRS[model]
        print("\n" + "#"*80)
        print(f" URUCHAMIANIE: MODEL '{model}' + POTOK '{pipeline}'")
        print("#"*80)

        # 1. Trening
        if not args.eval_only:
            print(f"\n[1/3] Uruchamianie treningu {model}...")
            train_model(
                model_name=model,
                pipeline_name=pipeline,
                epochs=args.epochs,
                batch_size=args.batch_size,
                resume=(not args.new),
                force_preprocess=args.preprocess
            )
        else:
            print(f"\n[1/3] Pomijanie treningu (tryb --eval_only)...")

        # Określenie ścieżki do wag
        model_path = f"models/{model}_{pipeline}_best.pth"
        if not os.path.exists(model_path):
            print(f"  [Ostrzeżenie] Brak specyficznych wag {model_path}, próba fallback do global_best_ecg_model.pth...")
            model_path = "models/global_best_ecg_model.pth"

        # 2. Ewaluacja - Zbiór główny (Zenodo)
        print(f"\n[2/3] Uruchamianie ewaluacji: Zenodo (CP-03)...")
        res_zenodo = evaluate_full_pipeline(
            model_ecg_path=model_path,
            record="CP-03",
            dataset="Zenodo",
            model_name=model,
            pipeline_name=pipeline
        )

        # 3. Ewaluacja - Zbiór alternatywny (IEEE)
        print(f"\n[3/3] Uruchamianie ewaluacji: IEEE (sub_6) [Generalizacja]...")
        res_ieee = evaluate_full_pipeline(
            model_ecg_path=model_path,
            record="sub_6",
            dataset="IEEE",
            model_name=model,
            pipeline_name=pipeline
        )

        # Zapisanie wyników
        results_report[model] = {
            'pipeline': pipeline,
            'zenodo': res_zenodo,
            'ieee': res_ieee
        }

    # 4. Generowanie raportu końcowego
    print("\n" + "="*80)
    print(" RAPORT KOŃCOWY Z EKSPERYMENTÓW")
    print("="*80)

    report_lines = [
        "# Zbiorcze Porównanie Eksperymentów Rekonstrukcji EKG",
        "",
        "Raport automatycznie wygenerowany po zakończeniu cyklu badawczego.",
        "",
        "## 1. Dokładność Rekonstrukcji EKG (Korelacja Pearsona r)",
        "",
        "| Model | Potok Preprocessingu | r (Zenodo - CP-03) | r (IEEE - sub_6) |",
        "| :--- | :--- | :--- | :--- |"
    ]

    for model, res in results_report.items():
        corr_zen = f"{res['zenodo']['mean_corr']:.4f}" if res['zenodo'] else "N/A"
        corr_ieee = f"{res['ieee']['mean_corr']:.4f}" if res['ieee'] else "N/A"
        report_lines.append(f"| `{model}` | `{res['pipeline']}` | **{corr_zen}** | **{corr_ieee}** |")

    report_lines.append("")
    report_lines.append("## 2. Błędy Wyznaczania Parametrów HRV (GT vs Predykcja)")
    report_lines.append("")

    for model, res in results_report.items():
        report_lines.append(f"### Model: `{model}` + Potok: `{res['pipeline']}`")
        report_lines.append("")
        report_lines.append("| Wskaźnik HRV | Zenodo (GT / Pred / diff) | IEEE (GT / Pred / diff) |")
        report_lines.append("| :--- | :--- | :--- |")

        # Wybór najważniejszych indeksów do tabeli porównawczej
        key_indices = ['HeartRate', 'MeanRR', 'SDNN', 'RMSSD', 'LF_HF_ratio', 'SD1_SD2_ratio']
        for key in key_indices:
            # Zenodo
            if res['zenodo']:
                gt_z = res['zenodo']['hrv_gt'].get(key, np.nan)
                pred_z = res['zenodo']['hrv_pred'].get(key, np.nan)
                diff_z = "N/A" if (np.isnan(gt_z) or np.isnan(pred_z)) else f"{abs(gt_z - pred_z):.2f}"
                val_z_str = f"{gt_z} / {pred_z} / {diff_z}"
            else:
                val_z_str = "N/A"

            # IEEE
            if res['ieee']:
                gt_i = res['ieee']['hrv_gt'].get(key, np.nan)
                pred_i = res['ieee']['hrv_pred'].get(key, np.nan)
                diff_i = "N/A" if (np.isnan(gt_i) or np.isnan(pred_i)) else f"{abs(gt_i - pred_i):.2f}"
                val_i_str = f"{gt_i} / {pred_i} / {diff_i}"
            else:
                val_i_str = "N/A"

            report_lines.append(f"| **{key}** | {val_z_str} | {val_i_str} |")
        report_lines.append("")

    report_content = "\n".join(report_lines)
    print(report_content)

    os.makedirs("final_results", exist_ok=True)
    report_path = "final_results/comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\n[SUKCES] Zapisano zbiorczy raport w '{report_path}'\n")

if __name__ == "__main__":
    main()

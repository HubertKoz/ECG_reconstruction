import os
import re
import shutil
import pandas as pd
import numpy as np
from evaluation.pipelines import evaluate_full_pipeline

# Definicje par model-potok
PAIRS = {
    'bilstm_transformer': 'kaisti',
    'tcn': 'wavelet',
    'unet1d': 'subband'
}

def main():
    models_dir = "models"
    output_base_dir = "results/checkpoints"
    os.makedirs(output_base_dir, exist_ok=True)

    # Wyszukiwanie wszystkich plików wag (.pth)
    if not os.path.exists(models_dir):
        print(f"[BŁĄD] Folder '{models_dir}' nie istnieje.")
        return

    pth_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    print(f"Znaleziono {len(pth_files)} plików wag w folderze '{models_dir}'.")

    results_list = []

    # Wyrażenia regularne do dopasowywania nazw plików wag
    # Np. unet1d_subband_checkpoint_100.pth, unet1d_subband_best.pth, unet1d_subband_final.pth
    checkpoint_pattern = re.compile(r"^([a-zA-Z0-9_]+)_([a-zA-Z0-9_]+)_checkpoint_(\d+)\.pth$")
    special_pattern = re.compile(r"^([a-zA-Z0-9_]+)_([a-zA-Z0-9_]+)_(best|final)\.pth$")

    for f in pth_files:
        model_name = None
        pipeline_name = None
        label = None

        m_cp = checkpoint_pattern.match(f)
        if m_cp:
            model_name, pipeline_name, epoch_str = m_cp.groups()
            label = f"epoch_{epoch_str}"
        else:
            m_sp = special_pattern.match(f)
            if m_sp:
                model_name, pipeline_name, spec_label = m_sp.groups()
                label = spec_label

        # Jeśli nazwa nie pasuje do wzorca lub para model-potok nie jest w definicjach, pomijamy
        if not model_name or not pipeline_name or model_name not in PAIRS or PAIRS[model_name] != pipeline_name:
            continue

        model_path = os.path.join(models_dir, f)
        print(f"\n================================================================================")
        print(f" EWALUACJA CHECKPOINTU: Model: {model_name} | Potok: {pipeline_name} | Label: {label}")
        print(f" Plik: {model_path}")
        print(f"================================================================================")

        # 1. Ewaluacja - Grupa Zenodo (CP-03, CP-09, CP-15)
        print(f"\n---> Ewaluacja: Grupa Zenodo (VHD)...")
        zenodo_results = []
        for rec in ["CP-03", "CP-09", "CP-15"]:
            try:
                res = evaluate_full_pipeline(
                    model_ecg_path=model_path,
                    record=rec,
                    dataset="Zenodo",
                    model_name=model_name,
                    pipeline_name=pipeline_name
                )
                if res:
                    zenodo_results.append(res)
            except Exception as e:
                print(f"[OSTRZEŻENIE] Błąd podczas ewaluacji Zenodo dla {rec}: {e}")

        # 2. Ewaluacja - Grupa IEEE (sub_6, sub_8, sub_9)
        print(f"\n---> Ewaluacja: Grupa IEEE (Healthy)...")
        ieee_results = []
        for rec in ["sub_6", "sub_8", "sub_9"]:
            try:
                res = evaluate_full_pipeline(
                    model_ecg_path=model_path,
                    record=rec,
                    dataset="IEEE",
                    model_name=model_name,
                    pipeline_name=pipeline_name
                )
                if res:
                    ieee_results.append(res)
            except Exception as e:
                print(f"[OSTRZEŻENIE] Błąd podczas ewaluacji IEEE dla {rec}: {e}")

        # Obliczenie średnich z grupy
        avg_zenodo_r = np.mean([r['mean_corr'] for r in zenodo_results]) if zenodo_results else np.nan
        avg_ieee_r = np.mean([r['mean_corr'] for r in ieee_results]) if ieee_results else np.nan

        # Zapisz wyniki liczbowe do listy
        row = {
            'file': f,
            'model': model_name,
            'pipeline': pipeline_name,
            'label': label,
            'zenodo_r': avg_zenodo_r,
            'ieee_r': avg_ieee_r,
        }

        # Wyciągamy wybrane wskaźniki HRV (średnie różnice bezwzględne dla grupy)
        key_indices = ['HeartRate', 'MeanRR', 'SDNN', 'RMSSD']
        for key in key_indices:
            # Zenodo
            zenodo_diffs = []
            for r in zenodo_results:
                if 'hrv_gt' in r and 'hrv_pred' in r:
                    gt = r['hrv_gt'].get(key, np.nan)
                    pred = r['hrv_pred'].get(key, np.nan)
                    if not (np.isnan(gt) or np.isnan(pred)):
                        zenodo_diffs.append(abs(gt - pred))
            row[f'zenodo_{key.lower()}_diff'] = np.mean(zenodo_diffs) if zenodo_diffs else np.nan

            # IEEE
            ieee_diffs = []
            for r in ieee_results:
                if 'hrv_gt' in r and 'hrv_pred' in r:
                    gt = r['hrv_gt'].get(key, np.nan)
                    pred = r['hrv_pred'].get(key, np.nan)
                    if not (np.isnan(gt) or np.isnan(pred)):
                        ieee_diffs.append(abs(gt - pred))
            row[f'ieee_{key.lower()}_diff'] = np.mean(ieee_diffs) if ieee_diffs else np.nan

        results_list.append(row)

        # 3. Kopiowanie wygenerowanych wykresów do dedykowanego podfolderu
        checkpoint_out_dir = os.path.join(output_base_dir, f"{model_name}_{pipeline_name}", label)
        os.makedirs(checkpoint_out_dir, exist_ok=True)

        temp_results_dir = "results"
        plots_to_move = []
        for rec in ["CP-03", "CP-09", "CP-15", "sub_6", "sub_8", "sub_9"]:
            plots_to_move.extend([
                f"reconstruction_quality_{rec}_0.png",
                f"poincare_{rec}_reconstructed.png",
                f"hrv_spectrum_{rec}_reconstructed.png"
            ])

        for plot_file in plots_to_move:
            src_path = os.path.join(temp_results_dir, plot_file)
            if os.path.exists(src_path):
                dest_path = os.path.join(checkpoint_out_dir, plot_file)
                try:
                    shutil.copy2(src_path, dest_path)
                    print(f"  -> Skopiowano wykres do {dest_path}")
                except Exception as e:
                    print(f"  -> Nie udało się skopiować wykresu {plot_file}: {e}")

    # 4. Zapisywanie tabelarycznego podsumowania
    if results_list:
        df_results = pd.DataFrame(results_list)
        
        # Sortowanie wierszy, aby label 'epoch_X' był uporządkowany numerycznie
        def sort_key(label_str):
            if label_str == 'best': return 999998
            if label_str == 'final': return 999999
            m = re.match(r'epoch_(\d+)', label_str)
            return int(m.group(1)) if m else 0

        df_results['sort_val'] = df_results['label'].apply(sort_key)
        df_results = df_results.sort_values(by=['model', 'sort_val']).drop(columns=['sort_val'])

        summary_csv_path = os.path.join(output_base_dir, "checkpoints_comparison.csv")
        df_results.to_csv(summary_csv_path, index=False)
        print(f"\n[SUKCES] Zapisano porównanie liczbowe checkpointów do: {summary_csv_path}")

        # Generowanie pliku Markdown z czytelnym zestawieniem (zgodnie z stylem plain text)
        summary_md_path = os.path.join(output_base_dir, "checkpoints_comparison.md")
        with open(summary_md_path, "w", encoding="utf-8") as f_md:
            f_md.write("# Zestawienie Wyników Checkpointów (Ewolucja Treningu)\n\n")
            f_md.write("Tabela przedstawia wyniki rekonstrukcji EKG oraz dokładności wyznaczania wskaźników HRV na różnych etapach uczenia uśrednione dla grup.\n\n")
            
            # Grupowanie po modelach
            for model, group in df_results.groupby('model'):
                f_md.write(f"## Model: {model} (Preprocess: {PAIRS[model]})\n\n")
                f_md.write("| Checkpoint (Label) | Średni r (Zenodo VHD) | Średni r (IEEE Healthy) | Średni HR Diff (Zenodo/IEEE) | Średni SDNN Diff (Zenodo/IEEE) |\n")
                f_md.write("| :--- | :---: | :---: | :---: | :---: |\n")
                
                for _, r in group.iterrows():
                    r_zen = f"{r['zenodo_r']:.4f}" if not np.isnan(r['zenodo_r']) else "N/A"
                    r_ieee = f"{r['ieee_r']:.4f}" if not np.isnan(r['ieee_r']) else "N/A"
                    
                    hr_zen = f"{r['zenodo_heartrate_diff']:.2f}" if not np.isnan(r['zenodo_heartrate_diff']) else "N/A"
                    hr_ieee = f"{r['ieee_heartrate_diff']:.2f}" if not np.isnan(r['ieee_heartrate_diff']) else "N/A"
                    hr_diff = f"{hr_zen} / {hr_ieee}"

                    sdnn_zen = f"{r['zenodo_sdnn_diff']:.2f}" if not np.isnan(r['zenodo_sdnn_diff']) else "N/A"
                    sdnn_ieee = f"{r['ieee_sdnn_diff']:.2f}" if not np.isnan(r['ieee_sdnn_diff']) else "N/A"
                    sdnn_diff = f"{sdnn_zen} / {sdnn_ieee}"

                    f_md.write(f"| {r['label']} | {r_zen} | {r_ieee} | {hr_diff} | {sdnn_diff} |\n")
                f_md.write("\n")
                
        print(f"[SUKCES] Zapisano raport porównawczy MD do: {summary_md_path}")

        # 5. Generowanie wykresów linii trendu dokładności dla epok
        import matplotlib.pyplot as plt
        
        for model, group in df_results.groupby('model'):
            epoch_rows = []
            for _, r in group.iterrows():
                m = re.match(r'epoch_(\d+)', r['label'])
                if m:
                    epoch_num = int(m.group(1))
                    epoch_rows.append({
                        'epoch': epoch_num,
                        'zenodo_r': r['zenodo_r'],
                        'ieee_r': r['ieee_r'],
                        'zenodo_sdnn': r['zenodo_sdnn_diff'],
                        'ieee_sdnn': r['ieee_sdnn_diff']
                    })
                elif r['label'] == 'final':
                    # Szacowanie numeru epoki dla final, domyślnie 200 (lub 150)
                    epoch_num = 150 if model == 'bilstm_transformer' else 200
                    epoch_rows.append({
                        'epoch': epoch_num,
                        'zenodo_r': r['zenodo_r'],
                        'ieee_r': r['ieee_r'],
                        'zenodo_sdnn': r['zenodo_sdnn_diff'],
                        'ieee_sdnn': r['ieee_sdnn_diff']
                    })

            if epoch_rows:
                df_epochs = pd.DataFrame(epoch_rows).drop_duplicates(subset=['epoch']).sort_values(by='epoch')
                
                if len(df_epochs) >= 1:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                    
                    # Wykres korelacji
                    ax1.plot(df_epochs['epoch'], df_epochs['zenodo_r'], marker='o', label='Grupa Zenodo (VHD) - r', color='crimson', linewidth=2)
                    ax1.plot(df_epochs['epoch'], df_epochs['ieee_r'], marker='s', label='Grupa IEEE (Zdrowi) - r', color='dodgerblue', linewidth=2)
                    ax1.set_ylabel('Korelacja Pearsona r')
                    ax1.set_title(f'Charakterystyka dokładności w epokach - Model: {model}')
                    ax1.grid(True, linestyle='--', alpha=0.6)
                    ax1.legend()
                    
                    # Wykres błędu SDNN
                    ax2.plot(df_epochs['epoch'], df_epochs['zenodo_sdnn'], marker='o', label='Grupa Zenodo (VHD) - SDNN Diff', color='darkred', linestyle='--', linewidth=2)
                    ax2.plot(df_epochs['epoch'], df_epochs['ieee_sdnn'], marker='s', label='Grupa IEEE (Zdrowi) - SDNN Diff', color='navy', linestyle='--', linewidth=2)
                    ax2.set_ylabel('Błąd SDNN (ms)')
                    ax2.set_xlabel('Epoka')
                    ax2.grid(True, linestyle='--', alpha=0.6)
                    ax2.legend()
                    
                    plt.tight_layout()
                    plot_path = os.path.join(output_base_dir, f"{model}_accuracy_epochs.png")
                    plt.savefig(plot_path, dpi=150)
                    plt.close()
                    print(f"  [+] Zapisano wykres dokładności epok dla {model} w '{plot_path}'")
    else:
        print("\n[Ostrzeżenie] Nie znaleziono żadnych pasujących checkpointów w folderze models.")

if __name__ == "__main__":
    main()

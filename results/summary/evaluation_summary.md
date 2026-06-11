# Wyniki Ewaluacji Modeli Rekonstrukcji EKG

Rekordy: ['CP-27', 'CP-50', 'UP-28', 'sub_19', 'sub_22', 'sub_26']

## bilstm_transformer  ·  pipeline: kaisti

| Label | Rekord | Zbiór | r | n_okien | HeartRate_diff | SDNN_diff | RMSSD_diff |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| best | CP-27 | zenodo | 0.5010 | 65 | 6.00 | 166.34 | 259.13 |
| best | CP-50 | zenodo | 0.2874 | 91 | 39.30 | 457.73 | 646.34 |
| best | UP-28 | zenodo | 0.2791 | 71 | 0.20 | 8.68 | 0.83 |
| best | sub_19 | ieee | 0.4822 | 111 | 1.20 | 28.84 | 54.28 |
| best | sub_22 | ieee | 0.0939 | 104 | 4.80 | 182.78 | 283.14 |
| best | sub_26 | ieee | 0.5345 | 112 | 0.20 | 20.39 | 37.67 |

## bilstm_transformer_pca  ·  pipeline: pca

| Label | Rekord | Zbiór | r | n_okien | HeartRate_diff | SDNN_diff | RMSSD_diff |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| best | CP-27 | zenodo | 0.8379 | 74 | 1.00 | 57.61 | 88.74 |
| best | CP-50 | zenodo | 0.1260 | 101 | 35.80 | 467.77 | 707.59 |
| best | UP-28 | zenodo | 0.1970 | 85 | 0.00 | 27.23 | 49.74 |
| best | sub_19 | ieee | 0.2168 | 111 | 5.10 | 143.88 | 250.14 |
| best | sub_22 | ieee | -0.0832 | 106 | 15.10 | 310.52 | 472.91 |
| best | sub_26 | ieee | 0.4979 | 141 | 6.30 | 302.20 | 420.79 |

## bilstm_transformer_v2  ·  pipeline: kaisti

| Label | Rekord | Zbiór | r | n_okien | HeartRate_diff | SDNN_diff | RMSSD_diff |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| best | CP-27 | zenodo | 0.1061 | 65 | 8.30 | 192.69 | 251.48 |
| best | CP-50 | zenodo | 0.1836 | 91 | 37.20 | 417.58 | 566.12 |
| best | UP-28 | zenodo | 0.3977 | 71 | 0.60 | 88.48 | 120.73 |
| best | sub_19 | ieee | 0.6042 | 111 | 0.90 | 34.15 | 69.50 |
| best | sub_22 | ieee | 0.2891 | 104 | 8.50 | 299.35 | 391.89 |
| best | sub_26 | ieee | 0.6513 | 112 | 0.10 | 9.41 | 18.92 |

## tcn  ·  pipeline: wavelet

| Label | Rekord | Zbiór | r | n_okien | HeartRate_diff | SDNN_diff | RMSSD_diff |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| best | CP-27 | zenodo | 0.3154 | 62 | 11.60 | 237.02 | 334.61 |
| best | CP-50 | zenodo | 0.0964 | 84 | 34.30 | 435.54 | 621.62 |
| best | UP-28 | zenodo | 0.3340 | 71 | 12.30 | 347.55 | 468.75 |
| best | sub_19 | ieee | 0.5024 | 111 | 2.90 | 75.23 | 141.62 |
| best | sub_22 | ieee | 0.0197 | 101 | 12.10 | 249.38 | 377.30 |
| best | sub_26 | ieee | 0.6631 | 138 | 1.50 | 96.55 | 154.52 |

## unet1d  ·  pipeline: subband

| Label | Rekord | Zbiór | r | n_okien | HeartRate_diff | SDNN_diff | RMSSD_diff |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| best | CP-27 | zenodo | 0.1134 | 65 | 24.50 | 361.21 | 528.42 |
| best | CP-50 | zenodo | 0.1748 | 91 | 37.50 | 412.95 | 585.48 |
| best | UP-28 | zenodo | 0.1084 | 71 | 3.00 | 205.63 | 230.02 |
| best | sub_19 | ieee | 0.5216 | 111 | 1.50 | 65.30 | 125.39 |
| best | sub_22 | ieee | 0.0502 | 104 | 15.50 | 278.11 | 375.14 |
| best | sub_26 | ieee | 0.3593 | 112 | 0.40 | 23.68 | 44.63 |

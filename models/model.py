import torch
import torch.nn as nn
import numpy as np


class ECGReconstructionModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, nhead=8, num_layers=4):
        super(ECGReconstructionModel, self).__init__()

        # Osobne enkodery BiLSTM dla PCG i SCG
        self.lstm_pcg = nn.LSTM(input_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.lstm_scg = nn.LSTM(input_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)

        # Po BiLSTM: hidden_dim * 2 kierunki * 2 sygnały = combined_dim
        combined_dim = hidden_dim * 2 * 2

        # Projekcja na wymiar Transformera
        self.feature_projection = nn.Linear(combined_dim, hidden_dim)

        # Wyuczalny positional encoding (max 2000 próbek)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2000, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dekoder MLP: rekonstrukcja amplitudy EKG w każdej próbce
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, pcg, scg):
        # pcg, scg: [batch, seq_len, 1]
        pcg_features, _ = self.lstm_pcg(pcg)
        scg_features, _ = self.lstm_scg(scg)

        combined = torch.cat((pcg_features, scg_features), dim=2)
        x = self.feature_projection(combined)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        return self.decoder(x)


if __name__ == "__main__":
    batch_size = 16
    seq_len = 250  # 1 sekunda przy fs=256 Hz
    model = ECGReconstructionModel()
    sample_pcg = torch.randn(batch_size, seq_len, 1)
    sample_scg = torch.randn(batch_size, seq_len, 1)
    out = model(sample_pcg, sample_scg)
    print(f"Wejście: {sample_pcg.shape}  →  Wyjście: {out.shape}")




    import torch.optim as optim
    import torch.nn as nn
    import os

    os.makedirs("models", exist_ok=True)

    # Konfiguracja sprzętu i modelu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGReconstructionModel().to(device)

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    # Funkcja straty MSE
    criterion = nn.MSELoss()



    # =========================================================================================
    # PRZYGOTOWANIE DANYCH PRZEZ PREPROCESSOR KAISTI
    # =========================================================================================
    loader = ECGDataLoader() 
    pre = Preprocessor(fs=256)

    # Wczytanie rekordu (Zenodo domyślnie, lub IEEE z adapterem 256Hz)
    signals_df = loader.load_zenodo(record='CP-01')
    if signals_df is None:
        print("Brak danych Zenodo, próba wczytania IEEE (resampled to 256Hz)...")
        signals_df = loader.load_ieee(record='sub_1', format=True)

    if signals_df is None:
        # Fallback na dummy jeśli nadal nie ma danych
        print("Brak danych, tworzenie danych syntetycznych.")
        signals_df = pd.DataFrame(np.random.randn(5000, 7), columns=['SCG_X', 'SCG_Y', 'SCG_Z', 'GCG_X', 'GCG_Y', 'GCG_Z', 'ECG_LA_RA'])

    # Uruchomienie zaawansowanego potoku (Kaisti 2018)
    results = pre.process_pipeline(signals_df)

    # Ekstrakcja znormalizowanych sygnałów (Model używa pcg i scg, mapujemy GCG na pcg dla fuzji)
    scg_channel = results['scg_final']
    pcg_channel = results['gcg_final']
    ecg_channel = results['ecg_final']

    # Przygotowanie maski do odrzucania okien z szumem
    clean_mask = results['clean_mask']
    fs = 256
    n_samples_epoch = int(results['epoch_sec'] * fs)

    n_samples_total = len(scg_channel)
    n_windows = n_samples_total // seq_len

    valid_scg, valid_pcg, valid_ecg = [], [], []

    # Tworzenie okien sekwencyjnych (seq_len) z pominięciem zanieczyszczonych epok
    for i in range(n_windows):
        start = i * seq_len
        end = start + seq_len
        
        epoch_start = start // n_samples_epoch
        epoch_end = (end - 1) // n_samples_epoch
        
        # Skiping if any part of the window touches a noisy epoch
        if epoch_start < len(clean_mask) and epoch_end < len(clean_mask):
            if clean_mask[epoch_start] and clean_mask[epoch_end]:
                valid_scg.append(scg_channel[start:end])
                valid_pcg.append(pcg_channel[start:end])
                valid_ecg.append(ecg_channel[start:end])

    # Formowanie wejść
    real_scg = torch.tensor(np.array(valid_scg), dtype=torch.float32).unsqueeze(-1)
    real_pcg = torch.tensor(np.array(valid_pcg), dtype=torch.float32).unsqueeze(-1)
    real_ecg = torch.tensor(np.array(valid_ecg), dtype=torch.float32).unsqueeze(-1)

    dataset = TensorDataset(real_pcg, real_scg, real_ecg)
    num_samples = len(dataset)
    print(f"Przygotowano {num_samples} fragmentów do treningu (odrzucono zaszumione).")

    # Wizualizacja jednej próbki przed treningiem
    if num_samples > 0:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.plot(real_ecg[0].squeeze().numpy(), label="Cel (EKG z-score)", color='red')
        plt.plot(real_scg[0].squeeze().numpy(), label="Wejscie (SCG z-score)", alpha=0.7)
        plt.plot(real_pcg[0].squeeze().numpy(), label="Wejscie (GCG z-score)", alpha=0.7)
        plt.title("Przykładowe wejście okna 1 sek. do modelu")
        plt.legend()
        plt.tight_layout()
        plt.show()
    # =========================================================================================

    # --- Sekcja do zakomentowania lub usunięcia po odkomentowaniu właściwego kodu powyżej ---
    # num_samples = 320 # Można tu wpisać większą wartość dla dokładniejszej weryfikacji
    # ...
    # -----------------------------------------------------------------------------------

    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Tworzenie DataLoaderów na potrzeby pętli
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)




    # --- Przykładowa pętla główna ---
    num_epochs = 50 # Dla celów szybkiego testu 10. Domyślnie było 50.
    best_corr = -1.0 # Zmienna do śledzenia najlepszego korelacją modelu

    print("Uruchamianie treningu...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_corr = validate(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Corr: {val_corr:.4f}")
        
        # Zapisywanie najlepszego modelu (najlepsza uzyskana korelacja na próbce walidacyjnej)
        if val_corr > best_corr:
            best_corr = val_corr
            torch.save(model.state_dict(), "models/best_ecg_model.pth")
            print(f" -> Zapisano nowy najlepszy model 'models/best_ecg_model.pth' (Korelacja: {best_corr:.4f})")
            
            # Opcjonalny warunek oparty o założenie z literatury
            if best_corr > 0.95:
                print(" -> [Info] Model po raz pierwszy osiągnął cel 0.95 korelacji!")

    # Pełne zapisanie ostatecznego kształtu wag modelu (zaraz po wyjściu z pętli roboczej)
    torch.save(model.state_dict(), "models/final_ecg_model.pth")
    print("Trening zakończony! Zapisano stan końcowy modelu w 'models/final_ecg_model.pth'.")
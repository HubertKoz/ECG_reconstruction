import torch
import torch.nn as nn

class HRVBeatDetectionModel(nn.Module):
    """
    Model sieci głębokiej dedykowany do detekcji pików uderzeń serca (Heart Beats)
    z sygnałów mechanicznych SCG i GCG. Zamiast odtwarzać amplitudę EKG (regresja),
    model ten klasyfikuje każdą próbkę czasu, przewidując prawdopodobieństwo [0, 1]
    wystąpienia piku uderzenia serca.
    """
    def __init__(self, input_dim=1, hidden_dim=128, nhead=8, num_layers=3):
        super(HRVBeatDetectionModel, self).__init__()
        
        # 1. Enkodery LSTM dla sekwencji czasowych (PCG/GCG i SCG)
        self.lstm_pcg = nn.LSTM(input_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.lstm_scg = nn.LSTM(input_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        
        # Wymiar po połączeniu obu BiLSTM: hidden_dim * 2 (kierunki) * 2 (modalności)
        combined_dim = hidden_dim * 2 * 2 
        
        # 2. Projekcja cech
        self.feature_projection = nn.Linear(combined_dim, hidden_dim)
        
        # Positional Encoding (uproszczony wyuczalny wektor, dopuszczający długie sekwencje do 4000 próbek)
        self.pos_embedding = nn.Parameter(torch.randn(1, 4000, hidden_dim)) 
        
        # 3. Transformer dla analizy globalnego kontekstu (odstępy między uderzeniami)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Dekoder mapujący wektor cech na prawdopodobieństwo bicia serca (Sigmoid)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Wyjście klasyfikacyjne per próbka
        )

    def forward(self, pcg, scg):
        # Wejście: [B, Seq_Len, 1]
        pcg_features, _ = self.lstm_pcg(pcg)
        scg_features, _ = self.lstm_scg(scg)
        
        # Scalenie ukrytych stanów
        combined = torch.cat((pcg_features, scg_features), dim=2)
        
        # Projekcja
        x = self.feature_projection(combined)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Mechanizm atencji
        x = self.transformer_encoder(x)
        
        # Prawdopodobieństwo pików
        output = self.decoder(x)
        return output

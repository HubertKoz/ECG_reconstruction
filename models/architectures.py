"""
Alternatywne architektury modeli rekonstrukcji EKG z sygnałów mechanicznych.

Wszystkie modele mają identyczny interfejs:
  forward(pcg, scg) -> output
  pcg, scg : Tensor [batch, seq_len, 1]
  output   : Tensor [batch, seq_len, 1]  (amplituda EKG per próbka)

Dostępne architektury:
  ECGReconstructionModel  – bazowy BiLSTM + Transformer (zaimportowany z model.py)
  CNNBiLSTMModel          – CNN feature extraction + BiLSTM temporal modeling
  TransformerOnlyModel    – czyste multi-head attention, sinusoidalne PE, bez LSTM
  TCNModel                – Temporal Convolutional Network (dilated causal conv, skip)
  ResNet1DModel           – 1D residual blocks + BiLSTM fusion decoder

ARCHITECTURE_REGISTRY: słownik nazwa → klasa (do importu w compare_all.py)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ECGReconstructionModel  # baseline


# ---------------------------------------------------------------------------
# 1. CNN + BiLSTM
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv1d → BN → ReLU → Conv1d → BN, z residual jeśli wymiary zgodne."""
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return F.relu(self.net(x) + self.skip(x))


class CNNBiLSTMModel(nn.Module):
    """
    Lokalne cechy wyodrębniane przez stos bloków Conv1D (z residual),
    globalne zależności czasowe modelowane przez BiLSTM.
    Dobre przy rekonstrukcji morfologii sygnału.
    """
    def __init__(self, input_dim=1, cnn_channels=64, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        # Osobne enkodery CNN dla PCG i SCG
        self.cnn_pcg = nn.Sequential(
            _ConvBlock(input_dim, 32, kernel=7),
            _ConvBlock(32, cnn_channels, kernel=5),
            _ConvBlock(cnn_channels, cnn_channels, kernel=5),
        )
        self.cnn_scg = nn.Sequential(
            _ConvBlock(input_dim, 32, kernel=7),
            _ConvBlock(32, cnn_channels, kernel=5),
            _ConvBlock(cnn_channels, cnn_channels, kernel=5),
        )

        # BiLSTM na złączonych cechach CNN
        self.bilstm = nn.LSTM(
            cnn_channels * 2, lstm_hidden,
            num_layers=lstm_layers, bidirectional=True,
            batch_first=True, dropout=0.2 if lstm_layers > 1 else 0.0
        )

        # Dekoder
        self.decoder = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, pcg, scg):
        # Wejście: [B, T, 1] → [B, 1, T] dla Conv1D
        p = self.cnn_pcg(pcg.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T, C]
        s = self.cnn_scg(scg.permute(0, 2, 1)).permute(0, 2, 1)

        fused, _ = self.bilstm(torch.cat([p, s], dim=2))
        return self.decoder(fused)


# ---------------------------------------------------------------------------
# 2. Transformer-Only (bez LSTM)
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    """Sinusoidalne positional encoding (Vaswani et al.), nieograniczona długość."""
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerOnlyModel(nn.Module):
    """
    Czyste multi-head self-attention bez LSTM.
    Sinusoidalne PE (nie wymaga trenowalnych embeddingów, działa dla każdej długości).
    Przetwarzanie obu sygnałów przez wspólny Transformer po liniowej projekcji.
    """
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        # Projekcja wejść: concatenate pcg+scg → [B, T, 2] → d_model
        self.input_proj = nn.Linear(input_dim * 2, d_model)
        self.pos_enc = _SinusoidalPE(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True  # Pre-LN: stabilniejszy trening
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, pcg, scg):
        x = torch.cat([pcg, scg], dim=2)     # [B, T, 2]
        x = self.input_proj(x)               # [B, T, d_model]
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.decoder(x)


# ---------------------------------------------------------------------------
# 3. TCN – Temporal Convolutional Network
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    """
    Jeden blok TCN: dilated causal conv -> BatchNorm -> ReLU -> Dropout -> residual.
    weight_norm usunieto - powoduje eksplozje NaN gdy ||v|| -> 0 przy dlugim treningu.
    Zastapiono BatchNorm + Kaiming init dla stabilnosci numerycznej.
    """
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation  # causal padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pad   = pad

        # Kaiming init - zapobiega eksplozji/zanikowi gradientu
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        # Causal: odcinamy nadmiarowe probki z prawej
        out = self.conv1(x)
        if self.pad:
            out = out[:, :, :-self.pad]
        out = F.relu(self.bn1(out))
        out = self.drop(out)
        out = self.conv2(out)
        if self.pad:
            out = out[:, :, :-self.pad]
        out = F.relu(self.bn2(out))
        out = self.drop(out)
        return F.relu(out + self.skip(x))


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network z wykładniczo rosnącymi dilacjami.
    Receptive field: kernel_size * sum(dilations) próbek.
    Nie wymaga LSTM — równolegle przetwarza całą sekwencję → szybki trening.
    """
    def __init__(self, input_dim=1, n_channels=64, kernel_size=3,
                 dilations=(1, 2, 4, 8, 16, 32), dropout=0.1):
        super().__init__()
        # Osobne TCN dla PCG i SCG
        def _build_tcn(in_d):
            layers = []
            in_ch = in_d
            for i, d in enumerate(dilations):
                out_ch = n_channels
                layers.append(_TCNBlock(in_ch, out_ch, kernel_size, d, dropout))
                in_ch = out_ch
            return nn.Sequential(*layers)

        self.tcn_pcg = _build_tcn(input_dim)
        self.tcn_scg = _build_tcn(input_dim)

        # Fusion + decoder
        self.fusion = nn.Conv1d(n_channels * 2, n_channels, 1)
        self.decoder = nn.Sequential(
            nn.Linear(n_channels, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, pcg, scg):
        # Conv1D oczekuje [B, C, T]
        p = self.tcn_pcg(pcg.permute(0, 2, 1))  # [B, n_ch, T]
        s = self.tcn_scg(scg.permute(0, 2, 1))
        fused = F.gelu(self.fusion(torch.cat([p, s], dim=1)))  # [B, n_ch, T]
        return self.decoder(fused.permute(0, 2, 1))            # [B, T, 1]


# ---------------------------------------------------------------------------
# 4. ResNet1D + BiLSTM
# ---------------------------------------------------------------------------

class _ResBlock1D(nn.Module):
    """1D residual block: Conv → BN → ReLU → Conv → BN + skip."""
    def __init__(self, channels, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel, padding=pad),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=pad),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class ResNet1DModel(nn.Module):
    """
    Encoder per-sygnał: Conv1D (stem) + stos bloków residualnych.
    Fuzja przez concatenation → BiLSTM dla kontekstu czasowego → dekoder.
    Połączenie lokalnych cech CNN i globalnych zależności LSTM.
    """
    def __init__(self, input_dim=1, res_channels=64, n_blocks=4, lstm_hidden=128):
        super().__init__()
        def _make_encoder():
            return nn.Sequential(
                nn.Conv1d(input_dim, res_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(res_channels),
                nn.ReLU(),
                *[_ResBlock1D(res_channels, kernel=7) for _ in range(n_blocks)]
            )

        self.encoder_pcg = _make_encoder()
        self.encoder_scg = _make_encoder()

        self.bilstm = nn.LSTM(
            res_channels * 2, lstm_hidden,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=0.2
        )
        self.decoder = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, pcg, scg):
        p = self.encoder_pcg(pcg.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T, C]
        s = self.encoder_scg(scg.permute(0, 2, 1)).permute(0, 2, 1)
        fused, _ = self.bilstm(torch.cat([p, s], dim=2))
        return self.decoder(fused)


# ---------------------------------------------------------------------------
# 5. 1D U-Net z połączeniami omijającymi (Skip Connections)
# ---------------------------------------------------------------------------

class UNet1DModel(nn.Module):
    """
    Zaawansowana architektura U-Net 1D z skip connections dedykowana do
    ciągłej rekonstrukcji morfologii EKG z sygnałów SCG/GCG.
    Obsługuje dynamiczny wymiar wejściowy (np. 1 dla standardu, 3 dla subband).
    """
    def __init__(self, input_dim=1, base_filters=32):
        super().__init__()

        # --- ENKODER (Analiza lokalna i kompresja) ---
        # Poziom 1: Blok wejściowy
        self.enc1_pcg = nn.Sequential(nn.Conv1d(input_dim, base_filters, 7, padding=3), nn.BatchNorm1d(base_filters), nn.ReLU())
        self.enc1_scg = nn.Sequential(nn.Conv1d(input_dim, base_filters, 7, padding=3), nn.BatchNorm1d(base_filters), nn.ReLU())

        # Poziom 2: Downsampling
        self.down1 = nn.MaxPool1d(2)
        self.enc2 = nn.Sequential(
            nn.Conv1d(base_filters * 2, base_filters * 4, 5, padding=2),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU()
        )

        # Poziom 3: Wąskie gardło (Bottleneck)
        self.down2 = nn.MaxPool1d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(base_filters * 4, base_filters * 8, 5, padding=2),
            nn.BatchNorm1d(base_filters * 8),
            nn.ReLU(),
            nn.Conv1d(base_filters * 8, base_filters * 4, 5, padding=2),
            nn.ReLU()
        )

        # --- DEKODER (Rekonstrukcja fali) ---
        self.up1 = nn.ConvTranspose1d(base_filters * 4, base_filters * 4, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv1d(base_filters * 8, base_filters * 2, 5, padding=2),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose1d(base_filters * 2, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv1d(base_filters * 4, base_filters, 5, padding=2),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.Conv1d(base_filters, 1, 3, padding=1)
        )

    def forward(self, pcg, scg):
        p = pcg.permute(0, 2, 1)
        s = scg.permute(0, 2, 1)

        e1_p = self.enc1_pcg(p)
        e1_s = self.enc1_scg(s)
        e1 = torch.cat([e1_p, e1_s], dim=1)

        e2_in = self.down1(e1)
        e2 = self.enc2(e2_in)

        b_in = self.down2(e2)
        b = self.bottleneck(b_in)

        u1 = self.up1(b)
        if u1.size(2) != e2.size(2):
            u1 = F.interpolate(u1, size=e2.size(2), mode='linear', align_corners=False)
        d1_in = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(d1_in)

        u2 = self.up2(d1)
        if u2.size(2) != e1.size(2):
            u2 = F.interpolate(u2, size=e1.size(2), mode='linear', align_corners=False)
        d2_in = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(d2_in)

        return d2.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# 6. ECGReconstructionModelV2 – BiLSTM + Transformer z sinusoidalnym PE
# ---------------------------------------------------------------------------

class ECGReconstructionModelV2(nn.Module):
    """
    Klon ECGReconstructionModel (v1) z jedyną zmianą:
      - pos_embedding (nn.Parameter) inicjalizowany sinusoidalnie zamiast losowo.

    Hipoteza: losowa inicjalizacja PE w v1 może zapamiętać strukturę czasową
    specyficzną dla pacjenta treningowego. Inicjalizacja sinusoidalna daje
    bardziej neutralny punkt startowy z deterministyczną strukturą pozycyjną.

    Dlaczego NIE fixed sinusoidal PE (register_buffer):
      Czyste sinusoidalne PE jest identyczne dla każdej próbki wejściowej.
      Na początku treningu uwaga Transformera jest zdominowana przez PE
      i staje się niezależna od treści LSTM → gradient do LSTM jest bliski
      zeru → model nie uczy się. nn.Parameter (nawet z init sinusoidalnym)
      dostaje gradient bezpośrednio i odblokowuje uczenie LSTM.

    Wszystko identyczne z v1 poza inicjalizacją PE.
    """
    def __init__(self, input_dim=1, hidden_dim=128, nhead=8, num_layers=4,
                 max_len=2000):
        super().__init__()

        self.lstm_pcg = nn.LSTM(input_dim, hidden_dim, num_layers=2, bidirectional=True,
                                batch_first=True, dropout=0.2)
        self.lstm_scg = nn.LSTM(input_dim, hidden_dim, num_layers=2, bidirectional=True,
                                batch_first=True, dropout=0.2)

        combined_dim = hidden_dim * 2 * 2
        self.feature_projection = nn.Linear(combined_dim, hidden_dim)

        # PE inicjalizowany sinusoidalnie, ale TRENOWALNY (nn.Parameter)
        # → dostaje gradient → odblokowuje uczenie LSTM
        # → startuje z deterministyczną strukturą zamiast losowego szumu
        pe = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2).float()
                        * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pos_embedding = nn.Parameter(pe.unsqueeze(0))  # [1, max_len, hidden_dim]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, pcg, scg):
        pcg_feat, _ = self.lstm_pcg(pcg)
        scg_feat, _ = self.lstm_scg(scg)
        combined = torch.cat((pcg_feat, scg_feat), dim=2)
        x = self.feature_projection(combined)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        return self.decoder(x)


# Rejestr architektur
# ---------------------------------------------------------------------------

ARCHITECTURE_REGISTRY = {
    'bilstm_transformer':     ECGReconstructionModel,
    'bilstm_transformer_pca': ECGReconstructionModel,   # alias — ta sama architektura, inny pipeline
    'cnn_bilstm':             CNNBiLSTMModel,
    'transformer_only':       TransformerOnlyModel,
    'tcn':                    TCNModel,
    'resnet1d':               ResNet1DModel,
    'unet1d':                 UNet1DModel,
    'bilstm_transformer_v2':  ECGReconstructionModelV2,
}


def count_parameters(model: nn.Module) -> int:
    """Zwraca liczbę trenowalnych parametrów modelu."""
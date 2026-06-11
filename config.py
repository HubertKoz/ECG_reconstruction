# Globalne stałe konfiguracyjne projektu.
# Importowanie stąd zamiast używania magicznych liczb (magic numbers) w kodzie.

# ── Sygnał ────────────────────────────────────────────────────────────────────
TARGET_FS: int = 256        # Docelowa częstotliwość próbkowania [Hz]
SEQ_LEN:   int = 1024       # Długość okna sekwencyjnego [próbki] = 4 s przy 256 Hz

# ── Podział train/val ─────────────────────────────────────────────────────────
SPLIT_SEED:  int   = 42     # Seed losowości podziału
VAL_RATIO:   float = 0.2    # Udział rekordów walidacyjnych (per źródło)
SPLIT_INFO_PATH: str = "data/cache/split_info.json"

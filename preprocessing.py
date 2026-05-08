from preprocessing import Preprocessor as NewPreprocessor

# Zachowanie nazwy Preprocessor dla kompatybilności wstecznej
Preprocessor = NewPreprocessor

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from data_loader import DataLoader
    
    # 1. Inicjalizacja loadera i załadowanie prawdziwych danych
    print("--- Test Kaisti Preprocessor na danych Zenodo ---")
    loader = DataLoader(base_data_dir="./data")

    # Ładujemy pierwszy rekord (np. CP-01)
    df_ieee = loader.load_zenodo()
    if df_ieee is not None:
        # 2. Uruchomienie preprocessingu Kaisti     
        fs = 256
        kp = Preprocessor(fs=fs)
        results = kp.process_pipeline(df_ieee)
        
        if results:
            print(f"Liczba wykrytych uderzeń (Envelope): {len(results['peaks_env'])}")
            print(f"Liczba wykrytych uderzeń (Morphological): {len(results['peaks_morph'])}")
            print("Zakończono preprocessing!")
        else:
            print("Błąd podczas przetwarzania potoku Kaisti.")
    else:
        print(f"Nie udało się załadować rekordu")
            print("Błąd podczas przetwarzania potoku Kaisti.")
    else:
        print(f"Nie udało się załadować rekordu")




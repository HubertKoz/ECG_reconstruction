from evaluation import evaluate_reconstruction_pipeline

def evaluate_and_plot(model_path='models/best_ecg_model.pth', record='CP-01', num_samples=3, base_data_dir='./data'):
    """
    Wrapper dla evaluate_reconstruction_pipeline z pakietu evaluation.
    """
    evaluate_reconstruction_pipeline(
        model_path=model_path, 
        record=record, 
        num_samples=num_samples, 
        base_data_dir=base_data_dir
    )

if __name__ == "__main__":
    evaluate_and_plot(model_path='models/global_best_ecg_model.pth', num_samples=3)

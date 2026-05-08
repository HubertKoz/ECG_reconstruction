from evaluation import evaluate_hrv_pipeline

def evaluate_record(record_name='CP-01', dataset='Zenodo', model_path='models/global_best_hr_model.pth', base_data_dir='./data'):
    """
    Wrapper dla evaluate_hrv_pipeline z pakietu evaluation.
    """
    evaluate_hrv_pipeline(
        record_name=record_name, 
        dataset=dataset, 
        model_path=model_path, 
        base_data_dir=base_data_dir
    )

if __name__ == "__main__":
    evaluate_record(record_name='CP-01', dataset='Zenodo')

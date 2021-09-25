from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
raw_data_dir = project_dir / 'data' / 'raw'
interim_data_dir = project_dir / 'data' / 'interim'
processed_data_dir = project_dir / 'data' / 'processed'
saved_model_dir = project_dir / 'data' / 'saved_models'
vis_dir = project_dir / 'plots'
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "Data"
OUTPUT_DIR_LOSS = PROJECT_DIR / "Outputs" / "Loss_data"
OUTPUT_DIR_FIGURES = PROJECT_DIR / "Outputs" / "Figures"
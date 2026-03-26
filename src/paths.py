from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR_LOSS = PROJECT_DIR / "outputs" / "Loss_data"
OUTPUT_DIR_FIGURES = PROJECT_DIR / "outputs" / "Figures"
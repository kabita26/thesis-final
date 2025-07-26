import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import load_and_filter_data
from src.feature_engineering import create_features
from src.model import train_model
import joblib

RAW_PATH = r"C:\Users\Rajan Mahato\OneDrive\Desktop\Thesis\data\raw\Nepal_Crop_main.csv"
PROCESSED_PATH = "data/processed/rice_data.csv"
MODEL_PATH = "outputs/models/rice_model.joblib"

def main():
    df = load_and_filter_data(RAW_PATH, PROCESSED_PATH)
    df = create_features(df)
    model, mse = train_model(df)
    print(f"Training MSE: {mse}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    main()

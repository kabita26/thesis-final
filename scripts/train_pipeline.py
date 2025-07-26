import sys
import os
import joblib

# Ensure UTF-8 encoding for safe printing (for Python 3.7+)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass  # For older Python versions

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preprocessing import load_and_filter_data
from src.feature_engineering import create_features
from src.model import train_model

# Paths C:\Users\Rajan Mahato\OneDrive - Softwarica College\Desktop\nepal-rice-yield-forecast\data
RAW_PATH = r"C:\Users\Rajan Mahato\OneDrive - Softwarica College\Desktop\nepal-rice-yield-forecast\data\raw\Nepal_Crop_main.csv"
PROCESSED_PATH = "data/processed/rice_data.csv"
MODEL_PATH = "outputs/models/rice_model.joblib"

def main():
    # Load and preprocess data
    if not os.path.exists(RAW_PATH):
        print(f"❌ File not found: {RAW_PATH}")
        return

    df = load_and_filter_data(RAW_PATH, PROCESSED_PATH)

    # Feature engineering
    df = create_features(df)

    # Train model
    model, mse = train_model(df)

    # Output training info
    print("\n✅ Model trained successfully")
    print(f"📉 Training MSE: {mse:.2f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()

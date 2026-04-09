# train.py

from config import DATA_PATH, MODEL_PATH, SCALER_PATH
from src.data_preprocessing import load_data, preprocess_data
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.utils import save_artifacts

def main():
    print("🚀 Loading data...")
    df = load_data(DATA_PATH)

    print("⚙️ Preprocessing...")
    X, y, scaler = preprocess_data(df)

    print("🧠 Feature Engineering...")
    X = feature_engineering(X)

    print("🤖 Training model...")
    model = train_model(X, y)

    print("💾 Saving model...")
    save_artifacts(model, scaler, MODEL_PATH, SCALER_PATH)

    print("✅ Training complete!")

if __name__ == "__main__":
    main()
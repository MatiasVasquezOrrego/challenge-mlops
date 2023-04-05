# packages for predictions
from .train_model import encode_dataset
import xgboost as xgb

# to read model, encoders and data 
import pickle
import pandas as pd

#packages for paths
import os
from pathlib import Path

#Logs
import logging
logging.basicConfig(level=logging.INFO)

def load_model(path: Path) -> xgb.XGBClassifier:
    with open(path / 'model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def load_encoders(path) -> list:
    with open(path / 'encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    category_encoder = encoders['category_encoder']
    label_encoders = encoders['label_encoders']
    columns_to_numpy = encoders['columns_to_numpy']
    return category_encoder, label_encoders, columns_to_numpy

def predict(features: pd.DataFrame) -> pd.DataFrame:
    # get models path
    main_path = Path(__file__).parent.parent.parent
    models_path = main_path / "models"
    
    # load model and encoders
    model = load_model(path=models_path)
    category_encoder, label_encoders, columns_to_numpy = load_encoders(path=models_path)

    encoded_features = encode_dataset(X=features, category_encoder=category_encoder, label_encoders=label_encoders, columns_to_numpy=columns_to_numpy)

    results = model.predict(encoded_features)
    logging.info(f"results: {results}")

    return results

if __name__ == "__main__":
    # get path to processed_data
    main_path = Path(__file__).parent.parent.parent
    processed_data_path = main_path / "data/processed/features.csv"

    # extract features
    features = pd.read_csv(processed_data_path)

    # make predictions
    results = predict(features=features)


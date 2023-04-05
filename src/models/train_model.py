#Modelos y metricas
import xgboost as xgb
from sklearn.metrics import classification_report

#Preprocesamiento
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from fast_ml.model_development import train_valid_test_split
import numpy as np
import pandas as pd

#Logs
import logging
logging.basicConfig(level=logging.INFO)

#Almacenamiento de modelo
import pickle

#packages for paths
import os
from pathlib import Path

def transform_to_numpy(X: pd.DataFrame, columns: list) -> np.array:
    return X.loc[:, columns].to_numpy()

def create_category_encoder(X_train: pd.DataFrame, y_train: pd.DataFrame, columns: list) -> ce.GLMMEncoder:
    category_encoder = ce.GLMMEncoder(binomial_target = True)
    category_encoder.fit(X_train[columns], y_train)
    return category_encoder

def encode_category_columns(X: pd.DataFrame, category_encoder: ce.GLMMEncoder) -> np.array:
    columns = category_encoder.feature_names_out_
    return category_encoder.transform(X[columns])

def create_label_encoders(columns_with_values: dict) -> dict:
    label_encoders = {}
    for column in columns_with_values.keys():
        le = LabelEncoder()
        le.fit(columns_with_values[column])
        label_encoders[column] = le
    return label_encoders

def encode_label_columns(X: pd.DataFrame, label_encoders: dict) -> np.array:
    label_encodes = np.array([])
    for column in label_encoders.keys():
        results = label_encoders[column].transform(X[column])
        results = results.reshape(results.shape[0], 1)
        if label_encodes.size == 0: 
            label_encodes = results
            continue
        label_encodes = np.concatenate((label_encodes, results), axis=1)
    return label_encodes

def encode_dataset(X: pd.DataFrame, category_encoder: ce.GLMMEncoder, label_encoders: dict, columns_to_numpy: list) -> np.array:
    not_encoded_variables = transform_to_numpy(X=X, columns=columns_to_numpy)
    label_encodes = encode_label_columns(X=X, label_encoders=label_encoders)
    category_encodes = encode_category_columns(X=X, category_encoder=category_encoder)
    return np.concatenate([not_encoded_variables, label_encodes, category_encodes], axis=1)

def get_encoders(X_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple:
    category_encoders_columns = ['Des-I', 'Emp-I', 'Des-O', 'Emp-O', 'TIPOVUELO', 'is_high_season']
    category_encoder = create_category_encoder(X_train=X_train, y_train=y_train, columns=category_encoders_columns)

    label_columns_with_values = {
        'DIANOM': ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"], 
        'day_period': ["morning", "afternoon", "evening"]
    }
    label_encoders = create_label_encoders(columns_with_values=label_columns_with_values)

    return category_encoder, label_encoders

def train(processed_data: pd.DataFrame, params: dict) -> tuple:
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(processed_data, target='delay_status', train_size=0.8, valid_size=0.1, test_size=0.1)

    category_encoder, label_encoders = get_encoders(X_train=X_train, y_train=y_train)
    columns_to_numpy = ['DIA', 'MES', 'AÑO']
    encoders = {}
    encoders['category_encoder'] = category_encoder
    encoders['label_encoders'] = label_encoders
    encoders['columns_to_numpy'] = columns_to_numpy

    X_train_encoded = encode_dataset(X=X_train, category_encoder=category_encoder, label_encoders=label_encoders, columns_to_numpy=columns_to_numpy)
    X_valid_encoded = encode_dataset(X=X_valid, category_encoder=category_encoder, label_encoders=label_encoders, columns_to_numpy=columns_to_numpy)
    X_test_encoded = encode_dataset(X=X_test, category_encoder=category_encoder, label_encoders=label_encoders, columns_to_numpy=columns_to_numpy)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_encoded, y_train)

    y_pred = model.predict(X_train_encoded)
    logging.info(f"train metrics:\n{classification_report(y_train, y_pred)}\n")

    y_pred = model.predict(X_valid_encoded)
    logging.info(f"valid metrics:\n{classification_report(y_valid, y_pred)}\n")

    y_pred = model.predict(X_test_encoded)
    logging.info(f"test metrics:\n{classification_report(y_test, y_pred)}\n")

    return model, encoders

if __name__ == "__main__":
    # get path to processed_data
    main_path = Path(__file__).parent.parent.parent
    processed_data_path = main_path / "data/processed/features.csv"

    # extract features
    features = pd.read_csv(processed_data_path)

    # define parameters
    params = {        
        'max_depth': 6, 
        'learning_rate': 0.1,
        'n_estimators': 100, 
        'gamma': 0, 
        'lambda': 0.1,
        'scale_pos_weight': 5,
        'subsample': 0.8,
        'max_leaves': 20,
        'min_child_weight': 0.9,
        'tree_method': 'hist'
    }
    logging.info(f"params: {params}")

    # train model
    model, encoders = train(processed_data=features, params=params)

    # save model and encoders
    model_path = main_path / "models/model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    encoders_path = main_path / "models/encoders.pkl"
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
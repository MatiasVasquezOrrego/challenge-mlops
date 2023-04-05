from typing import Dict
from fastapi import FastAPI, HTTPException

# features process
from src.features.build_features import preprocess_data, process_data 
from src.models.predict_model import predict
import pandas as pd

#Logs
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.post("/predict")
def predict_data(data: Dict) -> Dict:

    logging.info(f"data: {data}")

    if not data:
        raise HTTPException(status_code=400, detail="Empty input data")

    try:
        # process data
        raw_data = pd.DataFrame([data])
        preprocessed_data = preprocess_data(raw_data)
        processed_data = process_data(preprocessed_data)

        # Generate predictions
        prediction = predict(features=processed_data)
        logging.info(f"data: {prediction[0]}")

        # Return the predictions along with a success status response
        return {"prediction": str(prediction[0])}

    except Exception as e:
        # Return an error message along with a failure status response
        return {"status": "failure", "error": str(e)}

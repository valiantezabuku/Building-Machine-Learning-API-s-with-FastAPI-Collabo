from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd


app = FastAPI()


class SepsisFeatures(BaseModel):
    PRG: float
    PL: float
    PR: float
    SK: float
    TS: float
    M11: float
    BD2: float
    Age: float
    Insurance: float


@app.get('/')
def status_check():
    return {"Status": "API is online....."}


xgboost_pipeline = joblib.load("../models/xgboost.joblib")

random_forest_pipeline = joblib.load("../models/random_forest.joblib")

encoder = joblib.load("../models/encoder.joblib")


@app.post('/xgboost_prediction')
def predict_proba_sepsis(data: SepsisFeatures):
    
    df = pd.DataFrame([data.model_dump()])

    prediction = xgboost_pipeline.predict(df)

    pred_int = int(prediction[0])

    prediction = encoder.inverse_transform([pred_int])[0]

    # Get the probability of the predicted class
    probability = round(float(xgboost_pipeline.predict_proba(df)[0][pred_int] * 100),2)

    results = {"prediction": prediction, "probability": probability}

    return {"results": results}


@app.post('/random_forest_prediction')
def predict_proba_sepsis(data: SepsisFeatures):
    
    df = pd.DataFrame([data.model_dump()])

    prediction = random_forest_pipeline.predict(df)

    pred_int = int(prediction[0])

    prediction = encoder.inverse_transform([pred_int])[0]

    # Get the probability of the predicted class
    probability = round(float(random_forest_pipeline.predict_proba(df)[0][pred_int] * 100),2)

    results = {"prediction": prediction, "probability": probability}

    return {"results": results}

    
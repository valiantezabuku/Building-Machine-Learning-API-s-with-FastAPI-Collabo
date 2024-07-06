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


xgboost_pipeline = joblib.load("models/xgboost_pipeline.joblib")
enconder = joblib.load("models/enconer.joblib")


@app.post('/xgboost_prediction')
def predict_sepsis(data: SepsisFeatures):
    
    df = pd.DataFrame([data.model_dump()])

    prediction = xgboost_pipeline.predict(df)

    prediction = int(prediction[0])

    prediction = encoder.inverse_transform([prediction])[0]

    return {"prediction": prediction}
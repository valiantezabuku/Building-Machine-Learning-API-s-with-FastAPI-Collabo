import os
from dotenv import load_dotenv

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.coder import PickleCoder
from fastapi_cache.decorator import cache

from redis import asyncio as aioredis

from pydantic import BaseModel
from typing import Tuple, Dict, Union

from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.preprocessing._label import LabelEncoder
import joblib
import pandas as pd
from urllib.request import urlopen


from src.api.config import ONE_DAY_SEC, ONE_WEEK_SEC, XGBOOST_URL, RANDOM_FOREST_URL, ENCODER_URL, ENV_PATH

load_dotenv(ENV_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    url = os.getenv("REDIS_URL")
    username = os.getenv("REDIS_USERNAME")
    password = os.getenv("REDIS_PASSWORD")
    redis = aioredis.from_url(url=url, username=username,
                              password=password, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    yield


# FastAPI Object
app = FastAPI(
    title='Sepsis classification',
    version='0.0.1',
    description='Identify ICU patients at risk of developing sepsis',
    lifespan=lifespan,
)


# API input features
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


class Url(BaseModel):
    pipeline_url: str
    encoder_url: str


class ResultData(BaseModel):
    prediction: str
    probability: float


class PredictionResponse(BaseModel):
    execution_msg: str
    execution_code: int
    result: ResultData


class ErrorResponse(BaseModel):
    execution_msg: Union[str, None]
    execution_code: Union[int, None]
    result: Union[Dict[str, Union[str, int]], Union[Dict[str, None], None]]


# Load the model pipelines and encoder
# Cache for 1 day
@cache(expire=ONE_DAY_SEC, namespace='pipeline_resource', coder=PickleCoder)
async def load_pipeline(pipeline_url: Url, encoder_url: Url) -> Tuple[imbPipeline, LabelEncoder]:
    pipeline, encoder = None, None
    try:
        pipeline: imbPipeline = joblib.load(urlopen(pipeline_url))
        encoder: LabelEncoder = joblib.load(urlopen(encoder_url))
    except Exception:
        # Log exception
        pass
    finally:
        return pipeline, encoder


# Endpoints

# Status endpoint: check if api is online
@app.get('/')
@cache(expire=ONE_WEEK_SEC, namespace='status_check')  # Cache for 1 week
async def status_check():
    return {"Status": "API is online..."}


@cache(expire=ONE_DAY_SEC, namespace='pipeline_classifier')  # Cache for 1 day
async def pipeline_classifier(pipeline: imbPipeline, encoder: LabelEncoder, data: SepsisFeatures) -> ErrorResponse | PredictionResponse:
    output = ErrorResponse(**{'execution_msg': None,
                              'execution_code': None, 'result': None})
    try:
        # Create dataframe
        df = pd.DataFrame([data.model_dump()])

        # Make prediction
        prediction = pipeline.predict(df)

        pred_int = int(prediction[0])

        prediction = encoder.inverse_transform([pred_int])[0]

        # Get the probability of the predicted class
        probability = round(
            float(pipeline.predict_proba(df)[0][pred_int] * 100), 2)

        msg = 'Execution was successful'
        code = 1
        result = {"prediction": prediction, "probability": probability}

        output = PredictionResponse(
            **{'execution_msg': msg,
               'execution_code': code, 'result': result}
        )

    except Exception as e:
        msg = 'Execution failed'
        code = 0
        result = {'error': f"Omg, pipeline classsifier failure{e}"}
        output = ErrorResponse(**{'execution_msg': msg,
                                  'execution_code': code, 'result': result})

    finally:
        return output


# Xgboost endpoint: classify sepsis with xgboost
@app.post('/xgboost_prediction')
async def xgboost_classifier(data: SepsisFeatures) -> ErrorResponse | PredictionResponse:
    xgboost_pipeline, encoder = await load_pipeline(XGBOOST_URL, ENCODER_URL)
    output = await pipeline_classifier(xgboost_pipeline, encoder, data)
    return output


# Random forest endpoint: classify sepsis with random forest
@app.post('/random_forest_prediction')
async def random_forest_classifier(data: SepsisFeatures) -> ErrorResponse | PredictionResponse:
    random_forest_pipeline, encoder = await load_pipeline(RANDOM_FOREST_URL, ENCODER_URL)
    output = await pipeline_classifier(random_forest_pipeline, encoder, data)
    return output

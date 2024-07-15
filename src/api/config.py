import os

ONE_DAY_SEC = 24*60*60

ONE_WEEK_SEC = ONE_DAY_SEC*7

XGBOOST_URL = "https://raw.githubusercontent.com/valiantezabuku/Building-Machine-Learning-API-s-with-FastAPI-Collabo/develop/dev/models/xgboost.joblib"

RANDOM_FOREST_URL = "https://raw.githubusercontent.com/valiantezabuku/Building-Machine-Learning-API-s-with-FastAPI-Collabo/develop/dev/models/random_forest.joblib"

ENCODER_URL = "https://raw.githubusercontent.com/valiantezabuku/Building-Machine-Learning-API-s-with-FastAPI-Collabo/develop/dev/models/encoder.joblib"

BASE_DIR = './'  # Where Unicorn server runs from

ENV_PATH = os.path.join(BASE_DIR, 'src/api/.env2')

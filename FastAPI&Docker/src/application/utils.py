from pydantic import BaseModel, validator
from catboost import CatBoostRegressor

cat_list = [
    'sex',
    'dataset',
    'cp',
    'fbs',
    'restecg',
    'exang',
    'slope',
    'thal'
]

num_list = [
    'age',
    'trestbps',
    'chol',
    'thalch',
    'oldpeak',
    'ca'
]


class DataPayload(BaseModel):
    data: dict

    @validator('data', pre=True)
    def data_payload_check(cls, v):
        for col in num_list + cat_list:
            if col not in v:
                raise ValueError(f"{col} isn't in payload data.")
        return v
            
    class Config:
        schema_extra = {
            'example': {
                'data': {
                    'age': 57,
                    'sex': 'Male',
                    'dataset': 'Cleveland',
                    'cp': 'asymptomatic',
                    'trestbps': 140.0,
                    'chol': 192.0,
                    'fbs': False,
                    'restecg': 'normal',
                    'thalch': 148.0,
                    'exang': False,
                    'oldpeak': 0.4,
                    'slope': 'flat',
                    'ca': 0.0,
                    'thal': 'fixed defect',
                }
            }
        }


class PredictionResponse(BaseModel):
    class PredictionResponseData(BaseModel):
        prediction: float
    requestId: str
    data: PredictionResponseData


class RequestId:
    def __init__(self):
        self.id: str = None

    def set(self, id: str):
        self.id = id

    def get(self):
        return self.id


request_id = RequestId()

model = CatBoostRegressor()
model.load_model('./model/model.cbm')

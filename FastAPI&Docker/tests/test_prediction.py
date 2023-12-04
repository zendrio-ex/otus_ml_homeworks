import pytest
from src.application.routers import get_prediction
from src.application.utils import DataPayload


@pytest.mark.parametrize("data, result", [
    ({
        'age': 63,
        'sex': 'Male',
        'dataset': 'Cleveland',
        'cp': 'typical angina',
        'trestbps': 145.0,
        'chol': 233.0,
        'fbs': True,
        'restecg': 'lv hypertrophy',
        'thalch': 150.0,
        'exang': False,
        'oldpeak': 2.3,
        'slope': 'downsloping',
        'ca': 0.0,
        'thal': 'fixed defect'}, 0.8748),
    ({
        'age': 41,
        'sex': 'Female',
        'dataset': 'Cleveland',
        'cp': 'atypical angina',
        'trestbps': 130.0,
        'chol': 204.0,
        'fbs': False,
        'restecg': 'lv hypertrophy',
        'thalch': 172.0,
        'exang': False,
        'oldpeak': 1.4,
        'slope': 'upsloping',
        'ca': 0.0,
        'thal': 'normal'}, 0)])
def test_check_scoring(data, result):
    prediction = get_prediction(request=DataPayload(data=data))['data']['prediction']
    assert round(prediction, 4) == result

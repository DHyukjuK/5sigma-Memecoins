import pytest
from models.stage1_volatility import VolatilitySpikeClassifier
from models.stage1_yield import YieldTokenDemandRegressor
from data_pipeline.data_preprocessor import DataPreprocessor

@pytest.fixture
def integrated_data():
    # Mock processed data from all sources
    return {
        'X_vol': np.random.randn(50, 12, 8),
        'y_vol': np.random.randint(0, 2, 50),
        'X_yield': np.random.randn(50, 8),
        'y_yield': np.random.randn(50)
    }

def test_volatility_integration(integrated_data):
    model = VolatilitySpikeClassifier(input_shape=(12, 8))
    model.train(integrated_data['X_vol'], integrated_data['y_vol'])
    preds = model.predict_spike_probability(integrated_data['X_vol'][:1])
    assert 0 <= preds[0][0] <= 1

def test_yield_integration(integrated_data):
    model = YieldTokenDemandRegressor()
    model.train(integrated_data['X_yield'], integrated_data['y_yield'])
    pred = model.predict(integrated_data['X_yield'][:1])
    assert isinstance(pred[0], float)
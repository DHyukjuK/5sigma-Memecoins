import pytest
import numpy as np
from models.stage1_volatility import VolatilitySpikeClassifier

@pytest.fixture
def sample_data():
    X = np.random.randn(100, 12, 8)  # 100 samples, 12 timesteps, 8 features
    y = np.random.randint(0, 2, 100)
    return X, y

def test_model_initialization():
    model = VolatilitySpikeClassifier(input_shape=(12, 8))
    assert model.model is not None
    assert model.threshold == 0.65

def test_training(sample_data):
    X, y = sample_data
    model = VolatilitySpikeClassifier(input_shape=(12, 8))
    history = model.train(X, y, epochs=2)
    assert 'loss' in history.history
    assert history.epoch == [0, 1]

def test_prediction_shape(sample_data):
    X, _ = sample_data
    model = VolatilitySpikeClassifier(input_shape=(12, 8))
    preds = model.predict_spike_probability(X[:5])
    assert preds.shape == (5, 1)
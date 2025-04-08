import pytest
from models.stage2_allocation import PortfolioOptimizer
from trading.portfolio_env import PortfolioAllocationEnv

@pytest.fixture(scope="module")
def portfolio_env():
    env = PortfolioAllocationEnv()
    yield env
    env.close()

@pytest.fixture(scope="module")
def trained_allocator():
    allocator = PortfolioOptimizer()
    # Load pre-trained weights for testing
    allocator.model = PPO.load("tests/test_models/test_allocator.zip")
    yield allocator
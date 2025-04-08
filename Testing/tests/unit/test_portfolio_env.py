import pytest
import numpy as np
from trading.portfolio_env import PortfolioAllocationEnv

def test_env_initialization():
    env = PortfolioAllocationEnv()
    assert env.action_space.shape == (4,)
    assert env.observation_space.spaces['market_regime'].shape == (3,)

def test_reset():
    env = PortfolioAllocationEnv()
    obs = env.reset()
    assert isinstance(obs, dict)
    assert 'portfolio_state' in obs

def test_step():
    env = PortfolioAllocationEnv()
    env.reset()
    action = np.array([0.3, 0.2, 0.1, 0.4])
    obs, reward, done, info = env.step(action)
    assert not done
    assert isinstance(reward, float)
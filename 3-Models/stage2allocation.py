import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading.portfolio_env import PortfolioAllocationEnv
from typing import Dict

class PortfolioOptimizer:
    def __init__(self):
        self.env = DummyVecEnv([lambda: PortfolioAllocationEnv()])
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device='auto'
        )
    
    def train(self, total_timesteps=100000):
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict_allocation(self, regime_signals: Dict[str, float]) -> Dict[str, float]:
        obs = {
            "market_regime": np.array([
                regime_signals['volatility_spike'],
                regime_signals['yield_demand'],
                regime_signals['funding_shift']
            ]),
            "portfolio_state": np.array([0.25, 0.25, 0.25, 0.25])  # Initial equal allocation
        }
        action, _ = self.model.predict(obs, deterministic=True)
        return self._format_allocation(action)
    
    def _format_allocation(self, action: np.ndarray) -> Dict[str, float]:
        """Convert RL output to portfolio weights with constraints"""
        weights = {
            'PEPE_PERP': np.clip(action[0], 0, 0.5),
            'stETH_YT': np.clip(action[1], 0, 0.3),
            'ETH_PT': np.clip(action[2], 0, 0.4),
            'STABLES': np.clip(1 - sum(action[:3]), 0.2, 1.0)
        }
        # Normalize to ensure sum = 1
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    def save_model(self, path):
        self.model.save(path)
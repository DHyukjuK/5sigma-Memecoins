import gym
import numpy as np
from gym import spaces
from typing import Tuple, Dict

class PortfolioAllocationEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0.2], dtype=np.float32),
            high=np.array([0.5, 0.3, 0.4, 1.0], dtype=np.float32),
            shape=(4,)
        )
        
        self.observation_space = spaces.Dict({
            "market_regime": spaces.Box(-1, 1, shape=(3,)),
            "portfolio_state": spaces.Box(0, np.inf, shape=(4,))
        })
        
        self.reset()
    
    def reset(self):
        self.portfolio_value = 1.0  # Normalized initial value
        self.current_step = 0
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, dict]:
        # Normalize action to valid portfolio weights
        allocation = self._normalize_allocation(action)
        
        # Get asset returns based on current market regime
        returns = self._get_asset_returns()
        new_value = self.portfolio_value * (1 + np.dot(allocation, returns))
        
        # Calculate reward (Sharpe Ratio inspired)
        reward = self._calculate_reward(allocation, returns)
        
        # Update state
        self.portfolio_value = new_value
        self.current_step += 1
        
        return self._get_obs(), reward, False, {}
    
    def _get_obs(self) -> Dict:
        # Generate synthetic regime signals for training
        self.regime_signals = np.random.uniform(-1, 1, size=3)
        return {
            "market_regime": self.regime_signals,
            "portfolio_state": np.array([
                self.portfolio_value * 0.25,
                self.portfolio_value * 0.25,
                self.portfolio_value * 0.25,
                self.portfolio_value * 0.25
            ])
        }
    
    def _normalize_allocation(self, action: np.ndarray) -> np.ndarray:
        """Ensure valid portfolio allocation"""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action / np.sum(action)
    
    def _get_asset_returns(self) -> np.ndarray:
        """Generate asset returns based on current market regime"""
        # [PEPE_PERP, stETH_YT, ETH_PT, STABLES]
        volatility_effect = 0.1 * self.regime_signals[0]  # Higher volatility -> higher memecoin variance
        yield_effect = 0.05 * self.regime_signals[1]     # Yield demand boosts YT
        funding_effect = -0.02 * self.regime_signals[2]  # High funding -> negative perp returns
        
        base_returns = np.array([
            0.02 + volatility_effect + funding_effect,  # PEPE_PERP
            0.01 + yield_effect,                        # stETH_YT
            0.005 - 0.3 * volatility_effect,            # ETH_PT
            0.001                                       # STABLES
        ])
        
        # Add some noise
        return base_returns + np.random.normal(0, 0.01, size=4)
    
    def _calculate_reward(self, allocation: np.ndarray, returns: np.ndarray) -> float:
        """Sharpe Ratio inspired reward with risk penalty"""
        portfolio_return = np.dot(allocation, returns)
        risk_penalty = 0.5 * np.std(returns * allocation)
        return portfolio_return / (risk_penalty + 1e-9)  # Small epsilon to avoid division by zero
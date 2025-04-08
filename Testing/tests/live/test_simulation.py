import pytest
from datetime import datetime, timedelta
from main import run_simulation
from config.constants import TEST_ALLOCATIONS

class TestSimulation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.start_date = datetime.now() - timedelta(days=7)
        self.end_date = datetime.now()
    
    def test_historical_simulation(self):
        results = run_simulation(
            start=self.start_date,
            end=self.end_date,
            initial_balance=10000,
            test_mode=True
        )
        assert 'final_balance' in results
        assert results['final_balance'] > 0
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results

    def test_allocation_constraints(self):
        for alloc in TEST_ALLOCATIONS:
            assert sum(alloc.values()) == pytest.approx(1.0)
            assert 0 <= alloc['PEPE_PERP'] <= 0.5
            assert 0 <= alloc['stETH_YT'] <= 0.3
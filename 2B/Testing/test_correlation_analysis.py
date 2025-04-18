import pytest
import pandas as pd
from datetime import datetime, timedelta
from ..CorrelationAnalysis import HypeAnalyzer, analyze_sentiment

@pytest.fixture
def sample_data():
    """Generate test data for all components"""
    # Twitter test data
    twitter_data = pd.DataFrame({
        'date': [datetime.now() - timedelta(hours=i) for i in range(24)],
        'content': ['pepe is going to moon'] * 12 + ['pepe is scam'] * 12,
        'sentiment': [0.8] * 12 + [-0.5] * 12
    })
    
    # Reddit test data
    reddit_data = pd.DataFrame({
        'date': [datetime.now() - timedelta(hours=i) for i in range(24)],
        'content': ['buy $pepe now'] * 12 + ['sell pepe'] * 12,
        'sentiment': [0.9] * 12 + [-0.3] * 12
    })
    
    # Market test data
    price_data = pd.DataFrame({
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(24)],
        'close': [0.1 + 0.01*i for i in range(12)] + [0.22 - 0.005*i for i in range(12)],
        'volume': [1000 + 200*i for i in range(12)] + [3200 - 100*i for i in range(12)]
    }).set_index('timestamp')
    
    return twitter_data, reddit_data, price_data

def test_sentiment_analysis():
    """Test sentiment scoring"""
    test_df = pd.DataFrame({'content': [
        'pepe is amazing', 
        'this is terrible',
        'neutral comment'
    ]})
    result = analyze_sentiment(test_df)
    assert 'sentiment' in result.columns
    assert result['sentiment'][0] > 0  # Positive
    assert result['sentiment'][1] < 0  # Negative

@pytest.mark.asyncio
async def test_hype_analyzer(sample_data):
    """Test end-to-end analysis"""
    twitter, reddit, price = sample_data
    analyzer = HypeAnalyzer()
    results = await analyzer.analyze(twitter, reddit, price)
    
    # Verify output structure
    assert 'strong_signals' in results
    assert 'weak_signals' in results
    assert 'metrics' in results
    
    # Check specific metrics
    assert -1 <= results['metrics']['price_sentiment_corr'] <= 1
    assert results['metrics']['volume_spikes_count'] >= 0
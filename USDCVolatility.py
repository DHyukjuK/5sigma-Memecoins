import requests

def get_usdc_variance():
    # Get the 30-day price change percentage from CoinGecko
    url = "https://api.coingecko.com/api/v3/coins/usd-coin"
    response = requests.get(url)
    data = response.json()
    
    # Get volatility as percentage (convert to decimal)
    volatility_pct = data['market_data']['price_change_percentage_30d'] 
    volatility_decimal = volatility_pct / 100
    
    # Square to get variance
    variance = volatility_decimal ** 2
    
    return {
        '30_day_volatility_pct': volatility_pct,
        '30_day_variance': variance,
    }

result = get_usdc_variance()
print(f"USDC 30-Day Variance: {result['30_day_variance']:.8f}")
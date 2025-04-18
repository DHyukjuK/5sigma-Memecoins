import requests
from bs4 import BeautifulSoup

def get_shib_usd_average_volatility():
    url = "https://marketmilk.babypips.com/symbols/SHIBUSD/volatility?source=coinbase"
    
    try:
        # Set headers to mimic a browser visit
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for HTTP errors
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the average volatility element
        avg_element = soup.find('span', class_='_averageValue_1cq16_28')
        
        if not avg_element:
            raise ValueError("Could not find the average volatility element on the page")
        
        # Extract the precise value from the title attribute
        precise_avg = avg_element.get('title', '').replace('%', '')
        
        # Extract the displayed value (fallback if title isn't available)
        displayed_avg = avg_element.text.replace('%', '')
        
        # Use the precise value if available, otherwise use the displayed value
        avg_volatility = precise_avg if precise_avg else displayed_avg
        
        # Convert to float
        return float(avg_volatility)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Get and display the average volatility
avg_vol = get_shib_usd_average_volatility()
if avg_vol is not None:
    print(f"Average volatility for SHIB/USD over the last 3 months: {avg_vol:.4f}%")
else:
    print("Failed to retrieve volatility data.")
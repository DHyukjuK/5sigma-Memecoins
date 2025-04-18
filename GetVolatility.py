from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# Setup headless Chrome
options = Options()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.set_window_size(1920, 1080)

# Load the SHIB/USD volatility page
driver.get("https://marketmilk.babypips.com/symbols/SHIBUSD/volatility?source=coinbase")

# Dismiss any onboarding tooltips
while True:
    try:
        tooltip_button = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH,
                "//button[.//text()[contains(., 'Got it')] or .//text()[contains(., 'Explore')]]"
            ))
        )
        print(f"ℹ️ Dismissing tooltip: '{tooltip_button.text.strip()}'")
        driver.execute_script("arguments[0].click();", tooltip_button)
        time.sleep(0.5)
    except:
        print("✅ All tooltips dismissed.")
        break

# Wait for the Volatility Per Day section to be present
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "day"))
)

# Extract daily volatility value (from default timeframe shown)
soup = BeautifulSoup(driver.page_source, "html.parser")
daily_volatility = None
try:
    updated_span = soup.find("section", {"id": "day"}).find("span", class_="_averageValue_1cq16_28")
    if updated_span:
        daily_volatility = updated_span.get_text(strip=True)
        print("📊 Current Daily Volatility:", daily_volatility)
    else:
        print("❌ Volatility span not found.")
except Exception as e:
    print("❌ Error extracting volatility:", e)

# Done
driver.quit()
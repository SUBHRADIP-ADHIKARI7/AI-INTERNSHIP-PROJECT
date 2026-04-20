import requests
from bs4 import BeautifulSoup
import json

print("\n--- Testing LinkedIn Public Jobs API ---")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/114.0.0.0 Safari/537.36"
}
try:
    url = "https://www.linkedin.com/jobs/search?keywords=python+intern&location=India"
    res = requests.get(url, headers=headers, timeout=10)
    print("Status:", res.status_code)
    
    soup = BeautifulSoup(res.text, 'html.parser')
    
    # LinkedIn public job cards often have the class 'base-card' or 'base-search-card'
    job_cards = soup.find_all("div", class_="base-search-card__info")
    if not job_cards:
        # Fallback check for li elements
        job_cards = soup.find_all("li")
        
    print(f"Found {len(job_cards)} potential job elements.")
    
    # Try finding explicit job links
    links = soup.find_all("a", class_="base-card__full-link")
    print(f"Found {len(links)} job links.")
    
    if links:
        print("TITLE:", links[0].text.strip())
        print("LINK:", links[0].get("href"))
        
except Exception as e:
    print("Error:", e)

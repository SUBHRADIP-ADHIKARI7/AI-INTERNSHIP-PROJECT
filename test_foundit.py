import requests
from bs4 import BeautifulSoup
import json

print("=== TESTING FOUNDIT.IN ===")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
}
try:
    url = "https://www.foundit.in/srp/results?query=python+intern"
    res = requests.get(url, headers=headers, timeout=10)
    print("Status:", res.status_code)
    
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, 'html.parser')
        jobs = soup.find_all("div", class_="card-apply-content")
        print(f"Items found: {len(jobs)}")
        
        if jobs:
            title = jobs[0].find("div", class_="job-tittle").text.strip()
            print("SAMPLE:", title)
except Exception as e:
    print("Error:", e)

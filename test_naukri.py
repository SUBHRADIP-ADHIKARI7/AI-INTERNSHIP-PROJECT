import requests
from bs4 import BeautifulSoup

print("=== TESTING NAUKRI.COM ===")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
}

try:
    url = "https://www.naukri.com/python-intern-jobs?k=python%20intern"
    res = requests.get(url, headers=headers, timeout=10)
    print("Status:", res.status_code)
    
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, 'html.parser')
        articles = soup.find_all("article", class_="jobTuple")
        if not articles:
            # Maybe inside a different tag
            cards = soup.find_all("div", class_="srp-jobtuple-wrapper")
            print(f"Items found (srp-jobtuple-wrapper): {len(cards)}")
        else:
            print(f"Items found (jobTuple): {len(articles)}")
            
        print("Page Title:", soup.title.string if soup.title else "No Title")
        if "Access Denied" in res.text or "Cloudflare" in res.text:
            print("WARNING: Caught by Cloudflare.")
    else:
        print("Blocked/Error.")
except Exception as e:
    print("Error:", e)

import requests
from bs4 import BeautifulSoup
import urllib.parse
import json

print("\n=== TESTING NAUKRI VIA SEARCH ENGINE ===")
query = 'site:naukri.com/job-listings "python intern"'
encoded_query = urllib.parse.quote(query)

url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
}

try:
    res = requests.get(url, headers=headers, timeout=10)
    print("Status:", res.status_code)
    
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, 'html.parser')
        results = soup.find_all("a", class_="result__url")
        snippets = soup.find_all("a", class_="result__snippet")
        
        print(f"Found {len(results)} Naukri links!")
        
        for i, (link_el, snip_el) in enumerate(zip(results[:3], snippets[:3])):
            link = link_el.get("href", "")
            title = snip_el.text.strip()
            print(f"[{i+1}] {title[:60]}")
            print(f"    URL: {link[:100]}")
            
except Exception as e:
    print("Error:", e)

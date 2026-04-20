import requests
from bs4 import BeautifulSoup

print("=== TESTING DDG POST FOR NAUKRI ===")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/114.0.0.0 Safari/537.36",
    "Origin": "https://html.duckduckgo.com",
    "Referer": "https://html.duckduckgo.com/"
}
data = {
    "q": "python intern site:naukri.com/job-listings",
    "b": ""
}
try:
    res = requests.post("https://html.duckduckgo.com/html/", data=data, headers=headers, timeout=10)
    print("Status:", res.status_code)
    
    if res.status_code == 200:
        soup = BeautifulSoup(res.text, 'html.parser')
        results = soup.find_all("a", class_="result__url")
        snippets = soup.find_all("a", class_="result__snippet")
        
        print(f"Found {len(results)} Naukri links!")
        for i, (link_el, snip_el) in enumerate(zip(results[:3], snippets[:3])):
            print(f"[{i+1}] {snip_el.text.strip()[:60]}...")
            print(f"    URL: {link_el.get('href', '')[:100]}")
            
except Exception as e:
    print("Error:", e)

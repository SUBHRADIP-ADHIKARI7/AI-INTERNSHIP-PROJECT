import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

print("=== TESTING INDEED INDIA RSS ===")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/rss+xml, text/xml, application/xml"
}
try:
    url = "https://in.indeed.com/rss?q=python+intern&l=India"
    res = requests.get(url, headers=headers, timeout=10)
    print("Status:", res.status_code)
    
    if res.status_code == 200:
        print("RSS Body preview:", res.text[:200])
        try:
            root = ET.fromstring(res.text)
            channel = root.find("channel")
            items = channel.findall("item") if channel else []
            print(f"Found {len(items)} RSS items!")
            if items:
                print("SAMPLE TITLE:", items[0].find("title").text)
                print("SAMPLE LINK:", items[0].find("link").text)
        except Exception as parse_e:
            print("XML Parse Error:", parse_e)
    else:
        print("Blocked/Error. Body:", res.text[:200])
except Exception as e:
    print("Error:", e)

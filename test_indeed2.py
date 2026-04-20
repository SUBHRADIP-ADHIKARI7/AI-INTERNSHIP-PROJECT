import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

print("=== TEST 1: Indeed RSS with stronger headers ===")
headers = {
    "User-Agent": "python-feedparser/6.0.11 +https://github.com/kurtmckee/feedparser",
    "Accept": "text/xml, application/rss+xml, application/xml;q=0.9, */*;q=0.8",
}
try:
    url = "https://in.indeed.com/rss?q=python+intern&l=India&fromage=30"
    res = requests.get(url, headers=headers, timeout=10)
    print("Status:", res.status_code)
    ct = res.headers.get("Content-Type", "")
    print("Content-Type:", ct)
    if "xml" in ct or "<rss" in res.text[:100]:
        root = ET.fromstring(res.text)
        items = root.findall(".//item")
        print(f"RSS Jobs: {len(items)}")
        if items:
            print("TITLE:", items[0].find("title").text)
            print("LINK:", items[0].find("link").text)
    else:
        print("Got HTML (blocked). Body[:200]:", res.text[:200])
except Exception as e:
    print("Error:", e)

print("\n=== TEST 2: Indeed via feedparser library ===")
try:
    import feedparser
    d = feedparser.parse("https://in.indeed.com/rss?q=python+intern&l=India&fromage=30")
    print(f"FeedParser status: {d.get('status', 'N/A')}")
    print(f"Entries found: {len(d.entries)}")
    if d.entries:
        e = d.entries[0]
        print("TITLE:", e.title)
        print("LINK:", e.link)
except Exception as e:
    print("feedparser Error:", e)

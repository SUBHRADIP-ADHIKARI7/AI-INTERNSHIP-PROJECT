"""Quick test of scraping approaches for each platform"""
import requests
from bs4 import BeautifulSoup
import urllib.request
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ─── TEST 1: LINKEDIN ───────────────────────────────────────────────────────
print("\n=== LinkedIn ===")
try:
    url = "https://www.linkedin.com/jobs/search?keywords=python+intern&location=India"
    res = requests.get(url, headers=headers, timeout=10)
    print("Status:", res.status_code)
    soup = BeautifulSoup(res.text, 'html.parser')
    links = soup.find_all("a", class_="base-card__full-link")
    cards = soup.find_all("div", class_="base-search-card__info")
    print(f"Found {len(links)} job links, {len(cards)} cards")
    if links:
        print("SAMPLE LINK:", links[0].get("href", "")[:100])
    else:
        # Try alternate selectors
        all_links = soup.find_all("a", href=True)
        job_links = [a for a in all_links if "/jobs/view/" in a.get("href","")]
        print(f"Alternate /jobs/view/ links: {len(job_links)}")
        if job_links:
            print("SAMPLE:", job_links[0].get("href","")[:100])
except Exception as e:
    print("LinkedIn Error:", e)

# ─── TEST 2: X/TWITTER via Nitter (privacy frontend, publicly accessible) ───
print("\n=== X/Twitter via Nitter ===")
nitter_instances = [
    "https://nitter.poast.org",
    "https://nitter.privacydev.net",
    "https://nitter.net",
]
for nitter in nitter_instances:
    try:
        url = f"{nitter}/search?q=python+intern+hiring+india+apply&f=tweets"
        res = requests.get(url, headers=headers, timeout=8)
        print(f"{nitter} - Status: {res.status_code}")
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            tweets = soup.find_all("div", class_="tweet-content")
            urls_in_tweets = soup.find_all("a", class_="tweet-link")
            print(f"  Found {len(tweets)} tweets, {len(urls_in_tweets)} tweet links")
            if tweets:
                print("  SAMPLE TWEET:", tweets[0].text.strip()[:150])
            break
    except Exception as e:
        print(f"  {nitter} Error: {e}")

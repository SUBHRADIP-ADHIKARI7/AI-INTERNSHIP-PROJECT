"""Test reliable public APIs for Indian job market"""
import requests
from bs4 import BeautifulSoup
import json

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.5",
}

# Test 1: LinkedIn with correct public URL pattern
print("=== TEST: LinkedIn public jobs (correct URL) ===")
try:
    url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords=python+intern&location=India&start=0"
    res = requests.get(url, headers=headers, timeout=10)
    print("Status:", res.status_code, "| Len:", len(res.text))
    if res.status_code == 200 and len(res.text) > 500:
        soup = BeautifulSoup(res.text, 'html.parser')
        job_items = soup.find_all("li")
        print(f"Job list items: {len(job_items)}")
        links = soup.find_all("a", href=True)
        job_links = [a for a in links if "/jobs/view/" in a.get("href","")]
        print(f"Job view links: {len(job_links)}")
        if job_links:
            print("SAMPLE:", job_links[0].get("href","")[:150])
except Exception as e:
    print("Error:", e)

# Test 2: LinkedIn Guest API
print("\n=== TEST: LinkedIn Guest JSON API ===")
try:
    url = "https://www.linkedin.com/jobs/search?keywords=python+intern&location=India&trk=public_jobs_jobs-search-bar_search-submit"
    res = requests.get(url, headers=headers, timeout=10)
    print("Status:", res.status_code)
    soup = BeautifulSoup(res.text, 'html.parser')
    # Try different selectors
    for selector in ["job-search-card", "base-card", "result-card"]:
        items = soup.find_all("li", class_=lambda x: x and selector in x)
        print(f"  Selector '{selector}': {len(items)} items")
    # Try job-search-card__link
    links = soup.find_all("a", class_=lambda x: x and "job" in str(x).lower())
    print(f"  Job links: {len(links)}")
    if links:
        print("  SAMPLE:", links[0].get("href","")[:100])
except Exception as e:
    print("Error:", e)

# Test 3: Remotive API (remote jobs, already verified working)
print("\n=== TEST: Remotive API ===")
try:
    res = requests.get("https://remotive.com/api/remote-jobs?search=python&limit=3", timeout=8)
    print("Status:", res.status_code)
    data = res.json()
    jobs = data.get("jobs", [])
    print(f"Jobs found: {len(jobs)}")
    if jobs:
        print("SAMPLE title:", jobs[0].get("title"))
        print("SAMPLE url:", jobs[0].get("url"))
except Exception as e:
    print("Error:", e)

# Test 4: Adzuna API (has India jobs, free tier)
print("\n=== TEST: Adzuna Public ===")
try:
    res = requests.get("https://api.adzuna.com/v1/api/jobs/in/search/1?app_id=test&app_key=test&results_per_page=5&what=python+intern&content-type=application/json", timeout=8)
    print("Status:", res.status_code)
except Exception as e:
    print("Error:", e)

# Test 5: Dev.to job board
print("\n=== TEST: Dev.to job postings ===")
try:
    res = requests.get("https://dev.to/api/articles?tag=hiring&per_page=10", headers=headers, timeout=8)
    print("Status:", res.status_code)
    data = res.json()
    print(f"Articles: {len(data)}")
    if data:
        art = data[0]
        print("Title:", art.get("title"), "| URL:", art.get("url"))
except Exception as e:
    print("Error:", e)

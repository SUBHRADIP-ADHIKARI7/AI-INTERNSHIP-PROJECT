import requests
import json

print("=== TESTING NAUKRI.COM API ===")
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Safari/537.36",
    "appid": "109",
    "systemid": "109"
}
try:
    url = "https://www.naukri.com/jobapi/v3/search?noOfResults=10&urlType=search_by_keyword&searchType=adv&keyword=python+intern&pageNo=1"
    res = requests.get(url, headers=headers, timeout=10)
    print("Status:", res.status_code)
    try:
        data = res.json()
        jobs = data.get("jobDetails", [])
        print(f"Jobs API found: {len(jobs)}")
        if jobs:
            print("SAMPLE:", jobs[0].get("title"), "|", jobs[0].get("companyName"))
    except:
        print("Not JSON:", res.text[:200])
except Exception as e:
    print("Error:", e)

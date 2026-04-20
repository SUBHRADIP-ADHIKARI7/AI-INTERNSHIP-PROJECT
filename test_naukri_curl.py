from curl_cffi import requests

print("=== TESTING NAUKRI WITH CURL_CFFI ===")
try:
    url = "https://www.naukri.com/jobapi/v3/search?noOfResults=10&urlType=search_by_keyword&searchType=adv&keyword=python+intern&pageNo=1"
    headers = {
        "appid": "109",
        "systemid": "109",
        "clientid": "d3br",
    }
    # impersonate="chrome120" mimics the exact TLS/JA3/HTTP2 fingerprint of Chrome
    res = requests.get(url, headers=headers, impersonate="chrome120", timeout=10)
    print("Status:", res.status_code)
    
    if res.status_code == 200:
        data = res.json()
        jobs = data.get("jobDetails", [])
        print(f"Jobs JSON API found: {len(jobs)}")
        if jobs:
            print("SAMPLE:", jobs[0].get("title"), "|", jobs[0].get("companyName"))
except Exception as e:
    print("Error:", e)

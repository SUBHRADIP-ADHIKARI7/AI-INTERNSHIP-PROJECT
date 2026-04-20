import requests

print("=== TESTING NAUKRI MOBILE API ===")
headers = {
    "User-Agent": "Naukri_Android_App/2.0.0",
    "appid": "109",
    "systemid": "109",
    "ClientId": "d3br",
}
try:
    url = "https://www.naukri.com/jobapi/v3/search?noOfResults=10&urlType=search_by_keyword&searchType=adv&keyword=python+intern"
    res = requests.get(url, headers=headers, timeout=10)
    print("Status:", res.status_code)
    
    if res.status_code == 200:
        data = res.json()
        jobs = data.get("jobDetails", [])
        print(f"Jobs API found: {len(jobs)}")
        if jobs:
            print("SAMPLE:", jobs[0].get("title"), "|", jobs[0].get("companyName"))
            print("URL:", "https://www.naukri.com" + jobs[0].get("jdURL", ""))
except Exception as e:
    print("Error:", e)

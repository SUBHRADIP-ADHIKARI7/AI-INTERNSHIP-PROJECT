import requests

headers = {"User-Agent": "Mozilla/5.0"}

# ── Test 1: Jobicy API (free, aggregates Indeed + LinkedIn + Remotive) ──────
print("=== TEST: Jobicy API ===")
try:
    res = requests.get("https://jobicy.com/api/v2/remote-jobs?tag=python&biweekly=1&count=5", headers=headers, timeout=10)
    print("Status:", res.status_code)
    if res.status_code == 200:
        data = res.json()
        jobs = data.get("jobs", [])
        print(f"Jobs: {len(jobs)}")
        if jobs:
            j = jobs[0]
            print("Title:", j.get("jobTitle"))
            print("Company:", j.get("companyName"))
            print("URL:", j.get("url"))
except Exception as e:
    print("Jobicy Error:", e)

# ── Test 2: Arbeitnow API (free, global, indeed-like aggregation) ─────────────
print("\n=== TEST: Arbeitnow API ===")
try:
    res = requests.get("https://www.arbeitnow.com/api/job-board-api", headers=headers, timeout=10)
    print("Status:", res.status_code)
    if res.status_code == 200:
        data = res.json()
        jobs = data.get("data", [])
        print(f"Jobs: {len(jobs)}")
        if jobs:
            j = jobs[0]
            print("Title:", j.get("title"))
            print("Company:", j.get("company_name"))
            print("URL:", j.get("url"))
except Exception as e:
    print("Arbeitnow Error:", e)

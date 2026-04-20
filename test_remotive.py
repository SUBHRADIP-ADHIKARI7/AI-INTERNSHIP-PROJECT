import requests
import json

print("\n--- Testing Remotive API ---")
try:
    url = "https://remotive.com/api/remote-jobs?category=software-dev&search=junior"
    res = requests.get(url, timeout=10)
    data = res.json()
    jobs = data.get("jobs", [])
    if jobs:
        print(f"Found {len(jobs)} jobs. Top 2:")
        for job in jobs[:2]:
            print("TITLE:", job.get("title"))
            print("COMPANY:", job.get("company_name"))
            print("LINK:", job.get("url"))
    else:
        print("No jobs found on Remotive.")
except Exception as e:
    print("Remotive Error:", e)

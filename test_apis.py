import requests

print("--- Testing The Muse API ---")
try:
    url = "https://www.themuse.com/api/public/jobs?location=India&category=Computer%20and%20IT&level=Internship"
    res = requests.get(url, timeout=10)
    data = res.json()
    if "results" in data and len(data["results"]) > 0:
        for job in data["results"][:2]:
            print("TITLE:", job.get("name"))
            print("LINK:", job.get("refs", {}).get("landing_page"))
    else:
        print("No jobs found on The Muse India.")
except Exception as e:
    print("Muse Error:", e)

print("\n--- Testing JSearch (RapidAPI) ---")
try:
    rapidapi_key = "7b06b3a287mshbf44ca0eced0619p1e1274jsn9c5036a309d3"
    url = "https://jsearch.p.rapidapi.com/search"
    qs = {"query":"Python intern in India","page":"1","num_pages":"1"}
    headers = {
        "x-rapidapi-key": rapidapi_key,
        "x-rapidapi-host": "jsearch.p.rapidapi.com"
    }
    res = requests.get(url, headers=headers, params=qs, timeout=20)
    print("Status:", res.status_code)
    data = res.json()
    if data.get("data"):
        for job in data["data"][:2]:
            print("TITLE:", job.get("job_title"))
            print("LINK:", job.get("job_apply_link"))
    else:
        print("No JSearch data")
except Exception as e:
    print("JSearch Error:", e)

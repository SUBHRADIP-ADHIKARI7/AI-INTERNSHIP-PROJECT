import requests
import json
import os

rapidapi_key = "7b06b3a287mshbf44ca0eced0619p1e1274jsn9c5036a309d3"
url = "https://internships-api.p.rapidapi.com/search"
querystring = {"query": "python intern", "location": "India", "page": "1"}

headers = {
    "x-rapidapi-key": rapidapi_key,
    "x-rapidapi-host": "internships-api.p.rapidapi.com"
}

print("Querying API...")
try:
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    print("KEYS IN FIRST JOB:")
    if data.get("data") and len(data["data"]) > 0:
        job = data["data"][0]
        for k, v in job.items():
            if "link" in k or "url" in k or "apply" in k:
                print(f"  {k}: {v}")
    else:
        print("No data found or bad response:")
        print(json.dumps(data, indent=2))
except Exception as e:
    print("Error:", e)

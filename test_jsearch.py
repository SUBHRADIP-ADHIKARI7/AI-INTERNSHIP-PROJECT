import requests
import json
import os

rapidapi_key = "7b06b3a287mshbf44ca0eced0619p1e1274jsn9c5036a309d3"
url = "https://jsearch.p.rapidapi.com/search"

querystring = {"query":"Python intern in India","page":"1","num_pages":"1"}

headers = {
	"x-rapidapi-key": rapidapi_key,
	"x-rapidapi-host": "jsearch.p.rapidapi.com"
}

try:
    response = requests.get(url, headers=headers, params=querystring)
    print("Status:", response.status_code)
    data = response.json()
    if data.get("data"):
        job = data["data"][0]
        print("TITLE:", job.get("job_title"))
        print("COMPANY:", job.get("employer_name"))
        print("LINK:", job.get("job_apply_link"))
    else:
        print("No data:", data)
except Exception as e:
    print("Error:", e)

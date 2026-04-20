import requests, json

r = requests.get(
    "https://unstop.com/api/public/opportunity/search-result",
    params={
        "opportunity": "internships",
        "page": 1,
        "per_page": 3,
        "oppstatus": "open",
        "searchTerm": "python"
    },
    headers={"User-Agent": "Mozilla/5.0"}
)

data = r.json()["data"]["data"]
for j in data:
    org = j.get("organisation", {})
    print(f"Title: {j.get('title', '?')}")
    print(f"Org: {org.get('name', '?')}")
    print(f"URL: https://unstop.com/{j.get('public_url', '')}")
    print(f"Stipend: {j.get('stipend', {})}")
    print("---")

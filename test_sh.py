import requests
from bs4 import BeautifulSoup
import json

print("\n--- Testing SimplyHired India ---")
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/114.0.0.0 Safari/537.36"}
try:
    url = "https://www.simplyhired.co.in/search?q=intern&l=india"
    res = requests.get(url, headers=headers, timeout=10)
    print("Status:", res.status_code)
    soup = BeautifulSoup(res.text, 'html.parser')
    
    jobs = soup.find_all("li", class_="css-0")
    print("Found jobs:", len(jobs))
    for job in jobs[:2]:
        header = job.find("h3")
        if header and header.find("a"):
            title = header.find("a").text
            link = "https://www.simplyhired.co.in" + header.find("a")['href']
            print("TITLE:", title)
            print("LINK:", link)
except Exception as e:
    print("Error:", e)

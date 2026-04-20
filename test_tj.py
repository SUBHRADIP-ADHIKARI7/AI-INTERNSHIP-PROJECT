import requests
from bs4 import BeautifulSoup
import json

def test_timesjobs(keyword):
    print("Scraping TimesJobs India...")
    # Add 'intern' to keyword to enforce internship
    kw_encoded = f"{keyword} intern".replace(" ", "+")
    url = f"https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&searchText={kw_encoded}&txtLocation=India"
    
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        cards = soup.find_all("li", class_="clearfix job-bx wht-shd-bx")
        print(f"Found {len(cards)} jobs!")
        
        results = []
        for card in cards[:3]:
            # Title
            title_elem = card.find("h2").find("a")
            title = title_elem.text.strip() if title_elem else "Unknown"
            apply_link = title_elem['href'] if title_elem else "#"
            
            # Company
            company_elem = card.find("h3", class_="joblist-comp-name")
            company = company_elem.text.strip() if company_elem else "Unknown"
            # Cleanup company text
            if "(More Jobs)" in company:
                company = company.replace("(More Jobs)", "").strip()
                
            results.append({
                "title": title,
                "company": company,
                "link": apply_link
            })
            
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print("Error:", e)

test_timesjobs("python")

import requests
from bs4 import BeautifulSoup
import json

def test_scrape(keyword):
    print(f"Scraping Internshala for '{keyword}'...")
    url = f"https://internshala.com/internships/keywords-{keyword}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        print("Status:", res.status_code)
        
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Internshala cards are typically 'container-fluid individual_internship'
        cards = soup.find_all("div", class_="individual_internship")
        print(f"Found {len(cards)} cards on the page.")
        
        results = []
        for card in cards[:3]:
            # Title
            title_elem = card.find("h3", class_="job-internship-name")
            title = title_elem.text.strip() if title_elem else "Unknown Role"
            
            # Company
            company_elem = card.find("a", class_="link_display_like_text")
            company = company_elem.text.strip() if company_elem else "Unknown Company"
            
            # Link
            link = "https://internshala.com" + title_elem.find("a")['href'] if title_elem and title_elem.find("a") else "#"
            
            results.append({
                "title": title,
                "company": company,
                "link": link
            })
            
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print("Scrape error:", e)

test_scrape("python")

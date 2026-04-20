from playwright.sync_api import sync_playwright
import time

print("=== TESTING NAUKRI VIA PLAYWRIGHT ===")
try:
    with sync_playwright() as p:
        # Launch Chromium invisibly
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print("Navigating to Naukri (Python Intern jobs)...")
        page.goto("https://www.naukri.com/python-intern-jobs", timeout=60000)
        
        # Wait for job wrappers to load in the DOM
        page.wait_for_selector("div.srp-jobtuple-wrapper", timeout=15000)
        
        jobs = page.query_selector_all("div.srp-jobtuple-wrapper")
        print(f"Playwright found {len(jobs)} jobs!")
        
        if jobs:
            title_elem = jobs[0].query_selector("a.title")
            title = title_elem.inner_text() if title_elem else "N/A"
            link = title_elem.get_attribute("href") if title_elem else "N/A"
            
            company_elem = jobs[0].query_selector("a.comp-name")
            company = company_elem.inner_text() if company_elem else "N/A"
            
            print("SAMPLE JOB:", title)
            print("COMPANY:", company)
            print("LINK:", link)
            
        browser.close()
except Exception as e:
    print("Playwright Error:", e)

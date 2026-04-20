import time
from playwright.sync_api import sync_playwright

print("=== GRABBING NAUKRI HTML ===")
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://www.naukri.com/python-intern-jobs?k=python%20intern", timeout=60000)
        
        print("Waiting for page load...")
        time.sleep(6)  # Give JS time to build the list
        
        html_content = page.content()
        with open("naukri_dom.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print("DOM saved to naukri_dom.html!")
        browser.close()
except Exception as e:
    print("Error:", e)

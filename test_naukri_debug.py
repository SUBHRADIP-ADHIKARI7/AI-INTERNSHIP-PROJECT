from playwright.sync_api import sync_playwright
import time

print("=== CAPTURING NAUKRI DOM via PLAYWRIGHT ===")
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://www.naukri.com/python-intern-jobs", timeout=60000)
        
        print("Waiting 10 seconds for React to mount...")
        time.sleep(10)
        
        # Take screenshot to see if it's a captcha
        page.screenshot(path="naukri_screenshot.png")
        print("Screenshot saved to naukri_screenshot.png")
        
        # Print all divs with classes to find the job wrapper
        divs = page.query_selector_all("div[class]")
        classes = set()
        for d in divs[:50]:
            try:
                cls = d.get_attribute("class")
                if cls and "job" in cls.lower():
                    classes.add(cls)
            except:
                pass
        print("\nJob-related classes found in DOM:", classes)
        
        browser.close()
except Exception as e:
    print("Error:", e)

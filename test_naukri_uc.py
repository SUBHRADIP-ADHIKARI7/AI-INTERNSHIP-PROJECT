import time
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

print("=== TESTING NAUKRI WITH UNDETECTED CHROMEDRIVER ===")
try:
    options = uc.ChromeOptions()
    options.headless = True 
    # Try headless first, UC patches headless flags
    
    driver = uc.Chrome(options=options)
    driver.set_page_load_timeout(30)
    
    query = "python intern"
    print(f"Loading Naukri for {query}...")
    driver.get(f"https://www.naukri.com/{query.replace(' ', '-')}-jobs")
    
    # Wait for the main list wrapper
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.srp-jobtuple-wrapper"))
    )
    
    jobs = driver.find_elements(By.CSS_SELECTOR, "div.srp-jobtuple-wrapper")
    print(f"UC Chrome found {len(jobs)} jobs!")
    
    if jobs:
        title_el = jobs[0].find_element(By.CSS_SELECTOR, "a.title")
        comp_el = jobs[0].find_element(By.CSS_SELECTOR, "a.comp-name")
        print("SAMPLE:", title_el.text, "|", comp_el.text)
        print("LINK:", title_el.get_attribute("href"))

    driver.quit()
except Exception as e:
    print("UC Error:", e)
    try:
        driver.quit()
    except:
        pass

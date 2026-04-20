import os
import requests
from database import jobs_col  # This imports the bridge we just built!

def fetch_and_save_job(url):
    print(f"[SEARCH] Reading job from: {url}...")
    
    # We use the Jina Reader API to turn any URL into clean text for the AI
    # No API key needed for basic use!
    reader_url = f"https://r.jina.ai/{url}"
    
    try:
        response = requests.get(reader_url)
        if response.status_code == 200:
            job_data = {
                "url": url,
                "content": response.text,  # This is the full job description
                "status": "new"
            }
            # Save it to your MongoDB 'job_listings' collection
            jobs_col.insert_one(job_data)
            print("[SUCCESS] Job data saved to your Cloud Database!")
        else:
            print("[ERROR] Failed to read the page.")
    except Exception as e:
        print(f"[WARNING] Error: {e}")

# --- TEST IT OUT ---
# Paste a real internship link below to see it save to your cloud!
test_link = "https://r.jina.ai/https://en.wikipedia.org/wiki/Internship"
fetch_and_save_job(test_link)
import os
from database import jobs_col, resumes_col
from dotenv import load_dotenv

load_dotenv()

def find_my_matches():
    print("[INFO] AI Brain: Scanning your database for internships...")
    
    # 1. Fetch all the jobs you've scraped so far
    jobs = list(jobs_col.find())
    
    if not jobs:
        print("[WARNING] Your database is empty! Go run scraper.py first.")
        return

    print(f"[SUCCESS] Found {len(jobs)} jobs in your cloud.")
    
    for job in jobs:
        print(f"--- Analyzing Job URL: {job['url']} ---")
        # Tomorrow, we will add the AI Matching logic here!
        print("[STATUS] Waiting for OpenAI Key to 'read' this job...")

# --- EXECUTE ---
if __name__ == "__main__":
    find_my_matches()
from googlesearch import search
import time

print("=== TESTING GOOGLE SEARCH (NAUKRI AGGREGATION) ===")
try:
    query = 'site:naukri.com/job-listings "python intern" "posted"'
    print(f"Searching: {query}")
    
    results = []
    # advanced=True gets us title and description too!
    for j in search(query, num_results=5, advanced=True):
        results.append({
            "title": j.title,
            "url": j.url,
            "description": j.description
        })
        print(f"[{len(results)}] {j.title}")
        print(f"    {j.url}")
        
    print(f"Found {len(results)} Naukri jobs indexed by Google!")
except Exception as e:
    print("Error:", e)

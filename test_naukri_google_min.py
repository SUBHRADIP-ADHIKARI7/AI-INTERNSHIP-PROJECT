from googlesearch import search
print("=== TESTING GOOGLE SEARCH API (NAUKRI) ===")
try:
    urls = list(search('site:naukri.com/job-listings "python intern"', num_results=5))
    print(f"Found {len(urls)} URLs!")
    for u in urls:
        print("URL:", u)
except Exception as e:
    print("Error:", e)

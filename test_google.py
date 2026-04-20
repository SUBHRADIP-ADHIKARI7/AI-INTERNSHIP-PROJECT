from googlesearch import search
try:
    results = list(search("Python developer intern India apply", num_results=3, advanced=True))
    for r in results:
        print("TITLE:", getattr(r, 'title', 'No Title'))
        print("URL:", getattr(r, 'url', 'No URL'))
        print("DESC:", getattr(r, 'description', 'No Desc'))
        print("---")
except Exception as e:
    print("Error:", e)

import json
from tools import (scrape_linkedin, scrape_indeed, scrape_remotive,
                   scrape_freshershub, scrape_internshiphub, scrape_placementindia,
                   scrape_unstop, scrape_social_media)

def test_source(name, func, query="python"):
    print(f"\n--- Testing {name} ---")
    try:
        results = func(query, days_ago=7)
        print(f"Found {len(results)} results")
        if results:
            print(f"First result: {results[0].get('title')} @ {results[0].get('company')}")
            print(f"Link: {results[0].get('apply_link')[:80]}...")
        return len(results) > 0
    except Exception as e:
        print(f"Error in {name}: {e}")
        return False

if __name__ == "__main__":
    sources = [
        ("LinkedIn", scrape_linkedin),
        ("Indeed", scrape_indeed),
        ("Remotive", scrape_remotive),
        ("FreshersHub", scrape_freshershub),
        ("InternshipHub", scrape_internshiphub),
        ("PlacementIndia", scrape_placementindia),
        ("Unstop", scrape_unstop),
        ("Social Media (X/FB/IG)", scrape_social_media)
    ]
    
    for name, func in sources:
        test_source(name, func)

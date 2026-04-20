from tools import search_internships

print("=== FULL MULTI-SOURCE SCRAPER TEST ===\n")
result = search_internships.invoke("python developer intern")

# Parse the result
if "```internship_cards" in result:
    import json
    json_str = result.split("```internship_cards\n")[1].split("\n```")[0]
    cards = json.loads(json_str)
    print(f"Total cards found: {len(cards)}\n")
    for card in cards:
        print(f"  [{card['id']}] {card['title'][:60]}")
        print(f"       Company: {card['company']}")
        print(f"       Location: {card['location']}")
        print(f"       Link: {card['apply_link'][:100]}")
        print()
else:
    print("RAW:", result[:500])

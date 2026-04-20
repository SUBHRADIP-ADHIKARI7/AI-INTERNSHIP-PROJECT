from duckduckgo_search import DDGS
import json

res = list(DDGS().text("Python intern India apply open", max_results=3))
print(json.dumps(res, indent=2))

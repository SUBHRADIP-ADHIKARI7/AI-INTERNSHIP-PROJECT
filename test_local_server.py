import requests
import json

try:
    print("Testing keyword fallback interceptor on Local Server...")
    res = requests.post("http://127.0.0.1:5000/chat", json={"message": "now show me remotive.com jobs", "session_id": "test_intercept"}, timeout=15)
    print("Status:", res.status_code)
    try:
        data = res.json()
        print("Response received.")
    except Exception as je:
        print("Not json:", res.text)
except Exception as e:
    print("Error:", e)

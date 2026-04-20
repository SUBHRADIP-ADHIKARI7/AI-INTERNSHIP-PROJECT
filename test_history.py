import requests

try:
    print("Sending Turn 1...")
    res1 = requests.post("http://127.0.0.1:5000/chat", json={"message": "react intern", "session_id": "test_history_bug"}, timeout=30)
    print("Turn 1 Status:", res1.status_code)
    
    print("Sending Turn 2...")
    res2 = requests.post("http://127.0.0.1:5000/chat", json={"message": "now show me remotive.com jobs", "session_id": "test_history_bug"}, timeout=30)
    print("Turn 2 Status:", res2.status_code)
    if res2.status_code != 200:
        print("Turn 2 Error Output:", res2.text)
except Exception as e:
    print("Error:", e)

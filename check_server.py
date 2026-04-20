import urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:5000", timeout=2) as response:
        print("SUCCESS! Status:", response.status)
        print(response.read()[:200])
except Exception as e:
    print("FAILED TO CONNECT:", e)

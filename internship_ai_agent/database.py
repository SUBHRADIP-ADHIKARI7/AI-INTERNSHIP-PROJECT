import os
from pymongo import MongoClient
from dotenv import load_dotenv

# 1. Load the secrets
load_dotenv()

# 2. Setup the connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client["InternshipAgent"]

# 3. Define these OUTSIDE of any 'try' or 'if' blocks
# This ensures app.py can find them!
jobs_col = db["job_listings"]
resumes_col = db["user_resumes"]

# 4. Just a test print to confirm
try:
    client.admin.command('ping')
    print("[SUCCESS] Connection Successful: Your laptop is talking to MongoDB Atlas!")
except Exception as e:
    print(f"[ERROR] Connection Failed: {e}")
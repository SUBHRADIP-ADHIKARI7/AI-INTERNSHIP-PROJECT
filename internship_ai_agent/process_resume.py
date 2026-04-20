import PyPDF2
from database import resumes_col

def extract_and_upload_resume(pdf_path, user_name):
    # 1. Read the PDF
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    # 2. Create the User Document
    user_data = {
        "name": user_name,
        "resume_text": text,
        "vector_embedding": None, # Your AI Agent will fill this tomorrow
        "status": "active"
    }

    # 3. Save to MongoDB
    resumes_col.insert_one(user_data)
    print(f"[SUCCESS] {user_name}'s resume is now in the Cloud Database.")

if __name__ == "__main__":
    # Put your resume PDF file in the same folder and change the name here
    extract_and_upload_resume("my_resume.pdf", "John Doe")
# 🤖 AI Internship Finder Agent

An intelligent, conversational AI agent built with Flask, LangChain, and OpenAI that helps students and job seekers find their perfect internships. It features live resume parsing, real-time internship search via RapidAPI, and a real-time conversational streaming avatar powered by Akool.

## ✨ Features

- **Resume Parsing:** Upload a PDF resume, and the AI extracts your skills and experience.
- **AI Agent Chat:** Chat with a `gpt-3.5-turbo` LangChain agent to get career advice and internship recommendations.
- **Live Internship Data:** Uses the RapidAPI Internships API to fetch real, up-to-date internship postings worldwide.
- **Live Speaking Avatar:** Integrated with Akool Streaming Avatar and Agora RTC to provide a visually engaging, speaking AI assistant.
- **Beautiful UI:** A dark-themed, glassmorphism single-page frontend with smooth animations.

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.8+ installed.
- API keys for OpenAI, RapidAPI, and Akool.

### 2. Install Dependencies
Open your terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
You must provide your API keys for the integrations to work. Create a `.env` file in the root directory (where `app.py` is located) and add the following securely:

```env
# Required: For the AI to generate responses and parse resumes
# Note: Ensure your OpenAI account has active billing credits!
OPENAI_API_KEY=sk-your-openai-api-key-here

# Required: For fetching real internships (from RapidAPI)
RAPIDAPI_KEY=your-rapidapi-key-here

# Required: For the Live Avatar to initialize and speak
AKOOL_API_KEY=your-akool-api-key-here

# Optional: The ID of the specific Akool Avatar model you want to use
AKOOL_AVATAR_ID=default_avatar
```

### 4. Run the Server
Start the Flask application by running:
```bash
python app.py
```

### 5. Access the Web App
Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🛠️ Project Architecture

- `app.py`: The main Flask server. It exposes the web UI, `/chat` endpoint, `/upload-resume` parser, and `/api/avatar/session` for Akool session generation.
- `tools.py`: Contains the custom LangChain tool (`search_internships`) which contacts the RapidAPI to find jobs based on user queries.
- `templates/index.html`: The complete frontend. It handles the chat UI, Agora video streaming container, drag-and-drop resume uploading, and dynamic job card rendering using vanilla JavaScript and CSS.
- `requirements.txt`: Contains all vital Python dependencies (`flask`, `langchain`, `langchain-openai`, `requests`, `PyPDF2`, `langgraph`, etc.).

---

## 💡 Troubleshooting

- **`insufficient_quota` Error from OpenAI:** The chatbot will warn you if your OpenAI API key has run out of funds. Go to `platform.openai.com` to add credits.
- **Avatar Not Showing Up:** Ensure your `AKOOL_API_KEY` is correct. If the API key is missing or invalid, the app will gracefully skip the avatar initialization and just work as a text chatbot.
- **No Internships Found via API:** Ensure your `RAPIDAPI_KEY` is valid. The `tools.py` script has a safe fallback to a small mock database if it cannot connect to RapidAPI.

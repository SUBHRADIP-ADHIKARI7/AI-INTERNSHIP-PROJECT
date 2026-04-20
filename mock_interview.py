import re
import random
import io
import os
from flask import Blueprint, request, jsonify

mock_interview_bp = Blueprint('mock_interview', __name__, url_prefix='/api/interview')

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')

def get_resume_stream_from_req(req):
    session_id = req.form.get("session_id") if req.form else None
    if not session_id and req.is_json:
        session_id = req.json.get("session_id")
    if session_id:
        filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}.pdf")
        if os.path.exists(filepath):
            return open(filepath, 'rb')
    if 'resume' in req.files:
        return req.files['resume'].stream
    return None

# ─────────────────────────────────────────────────────────────────────
# Comprehensive Tech Skills Dictionary (canonical name → search terms)
# ─────────────────────────────────────────────────────────────────────
SKILLS_DB = {
    # Languages
    "Python": ["python"], "JavaScript": ["javascript", "js "], "TypeScript": ["typescript", "ts "],
    "Java": ["java ","java,","java."], "C++": ["c++","cpp"], "C#": ["c#","csharp","c sharp"],
    "Go": ["golang"," go "], "Rust": [" rust ","rust,"], "Ruby": [" ruby ","ruby,"],
    "PHP": [" php ","php,"], "Swift": [" swift ","swift,"], "Kotlin": ["kotlin"],
    "Dart": [" dart ","dart,"], "Scala": ["scala"], "R": [" r ","r,","r programming"],
    "MATLAB": ["matlab"], "Perl": [" perl ","perl,"], "Shell": ["shell","bash","zsh","powershell"],
    # Frontend
    "React": ["react","reactjs","react.js"], "Angular": ["angular"],
    "Vue.js": ["vue","vuejs","vue.js"], "Next.js": ["next.js","nextjs"],
    "Svelte": ["svelte"], "jQuery": ["jquery"], "HTML": ["html","html5"],
    "CSS": ["css","css3"], "SASS": ["sass","scss"], "Tailwind CSS": ["tailwind"],
    "Bootstrap": ["bootstrap"], "Material UI": ["material ui","mui"],
    # Backend / Frameworks
    "Node.js": ["node.js","nodejs","node js"], "Express": ["express","expressjs"],
    "Django": ["django"], "Flask": ["flask"], "FastAPI": ["fastapi","fast api"],
    "Spring Boot": ["spring boot","spring"], "Laravel": ["laravel"],
    "Ruby on Rails": ["rails","ruby on rails"], "ASP.NET": ["asp.net",".net","dotnet"],
    "NestJS": ["nestjs","nest.js"],
    # Databases
    "SQL": [" sql ","sql,"], "MySQL": ["mysql"], "PostgreSQL": ["postgresql","postgres"],
    "MongoDB": ["mongodb","mongo"], "Redis": ["redis"], "SQLite": ["sqlite"],
    "Oracle DB": ["oracle"], "DynamoDB": ["dynamodb"], "Cassandra": ["cassandra"],
    "Firebase": ["firebase"], "Supabase": ["supabase"],
    # Cloud / DevOps
    "AWS": [" aws ","aws,","amazon web services"], "Azure": ["azure"],
    "GCP": [" gcp ","google cloud"], "Docker": ["docker"],
    "Kubernetes": ["kubernetes","k8s"], "Terraform": ["terraform"],
    "Jenkins": ["jenkins"], "CI/CD": ["ci/cd","ci cd","cicd"],
    "Ansible": ["ansible"], "Nginx": ["nginx"], "Apache": ["apache"],
    "Linux": ["linux"], "Git": [" git ","git,","github","gitlab","bitbucket"],
    # Data / ML / AI
    "Machine Learning": ["machine learning"," ml ","ml,"], "Deep Learning": ["deep learning"],
    "TensorFlow": ["tensorflow"], "PyTorch": ["pytorch"],
    "Keras": ["keras"], "Scikit-learn": ["scikit-learn","sklearn","scikit learn"],
    "Pandas": ["pandas"], "NumPy": ["numpy"], "NLP": [" nlp ","nlp,","natural language processing"],
    "Computer Vision": ["computer vision","opencv","cv "], "Data Science": ["data science"],
    "Data Analytics": ["data analytics","data analysis"], "Power BI": ["power bi","powerbi"],
    "Tableau": ["tableau"], "Apache Spark": ["apache spark"," spark "],
    "Hadoop": ["hadoop"], "Airflow": ["airflow"],
    # Mobile
    "React Native": ["react native"], "Flutter": ["flutter"],
    "Android": ["android"], "iOS": [" ios ","ios,","swiftui"],
    "Kotlin Multiplatform": ["kotlin multiplatform"],
    # APIs / Protocols
    "REST API": ["rest api","restful","rest "], "GraphQL": ["graphql"],
    "gRPC": ["grpc"], "WebSocket": ["websocket","socket.io"],
    # Testing
    "Jest": ["jest"], "Mocha": ["mocha"], "Cypress": ["cypress"],
    "Selenium": ["selenium"], "pytest": ["pytest"], "JUnit": ["junit"],
    # Tools
    "Jira": ["jira"], "Figma": ["figma"], "Postman": ["postman"],
    "VS Code": ["vs code","vscode"], "IntelliJ": ["intellij"],
    # Blockchain / Emerging
    "Blockchain": ["blockchain"], "Solidity": ["solidity"], "Web3": ["web3"],
    "Ethereum": ["ethereum"],
    # Soft Skills that matter for interviews
    "Agile": ["agile","scrum","kanban"], "Leadership": ["leadership","team lead"],
    "Communication": ["communication"], "Problem Solving": ["problem solving","problem-solving"],
}

ROLE_MAP = {
    "Frontend Developer": ["react","angular","vue","html","css","next.js","tailwind","svelte","jquery","bootstrap","material ui","sass"],
    "Backend Developer": ["node.js","express","django","flask","fastapi","spring boot","laravel","nestjs","asp.net"],
    "Full Stack Developer": ["react","node.js","express","django","mongodb","postgresql","next.js"],
    "Data Scientist": ["machine learning","deep learning","tensorflow","pytorch","pandas","numpy","scikit-learn","data science","r"],
    "Data Analyst": ["sql","data analytics","tableau","power bi","pandas","excel","python"],
    "ML Engineer": ["machine learning","deep learning","tensorflow","pytorch","keras","nlp","computer vision"],
    "DevOps Engineer": ["docker","kubernetes","aws","terraform","jenkins","ci/cd","linux","ansible","nginx"],
    "Mobile Developer": ["react native","flutter","android","ios","kotlin","swift","dart"],
    "Cloud Engineer": ["aws","azure","gcp","docker","kubernetes","terraform"],
    "UI/UX Designer": ["figma","css","html","bootstrap","material ui","tailwind"],
    "Blockchain Developer": ["blockchain","solidity","web3","ethereum"],
    "Software Engineer": ["python","javascript","java","c++","git","sql","rest api"],
}


def extract_text_from_pdf(file_stream):
    """Extract text from PDF using pdfminer.six, with PyPDF2 + (optional) OCR fallback."""
    text = ""

    # ── Attempt 1: pdfminer.six (best for text-based PDFs) ──
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        file_stream.seek(0)
        text = pdfminer_extract(file_stream)
    except Exception as e:
        print(f"[pdfminer] extraction failed: {e}")

    # ── Attempt 2: PyPDF2 fallback ──
    if not text or len(text.strip()) < 50:
        try:
            import PyPDF2
            file_stream.seek(0)
            reader = PyPDF2.PdfReader(file_stream)
            text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        except Exception as e:
            print(f"[PyPDF2] extraction failed: {e}")

    # ── Attempt 3: OCR fallback (for scanned PDFs) ──
    if not text or len(text.strip()) < 50:
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            file_stream.seek(0)
            images = convert_from_bytes(file_stream.read())
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img) + "\n"
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        except ImportError:
            print("[OCR] pytesseract/pdf2image not installed — skipping OCR fallback")
        except Exception as e:
            print(f"[OCR] extraction failed: {e}")

    return text.strip()


def extract_skills(text):
    """Match resume text against the curated skills dictionary. Returns list of canonical skill names."""
    text_lower = " " + text.lower().replace("\n", " ").replace("\t", " ") + " "
    found = []
    for canonical_name, keywords in SKILLS_DB.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(canonical_name)
                break
    return found


def infer_role(skills):
    """Infer the best matching role from extracted skills using weighted matching."""
    skills_lower = set(s.lower() for s in skills)
    scores = {}
    for role, role_keywords in ROLE_MAP.items():
        count = sum(1 for kw in role_keywords if kw in skills_lower)
        if count > 0:
            scores[role] = count
    if not scores:
        return "Software Engineer"
    return max(scores, key=scores.get)


@mock_interview_bp.route('/upload-resume', methods=['POST'])
def parse_resume():
    stream = get_resume_stream_from_req(request)
    if not stream:
        return jsonify({"error": "No resume found. Please provide session_id or file."}), 400

    # Extract text using multi-method approach
    text = extract_text_from_pdf(stream)

    if not text or len(text) < 30:
        return jsonify({"error": "Could not extract text from the PDF. Please ensure it is a text-based or clearly scanned resume."}), 400

    # Extract skills using curated dictionary matching
    found_skills = extract_skills(text)

    if not found_skills:
        found_skills = ['Communication', 'Problem Solving', 'Leadership']

    # Infer role
    inferred_role = infer_role(found_skills)

    return jsonify({
        "skills": found_skills,
        "role": inferred_role,
        "parsedText": text[:800] + ("..." if len(text) > 800 else ""),
        "skillCount": len(found_skills)
    })

import json
import random
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

def mock_generate_questions(role, skills, difficulty='medium'):
    # Default skills if none found
    pool = skills if skills and len(skills) > 0 else ['Core Technologies', 'System Architecture', 'Problem Solving']
    
    coding_instruction = "\nCRITICAL: EXACTLY 3 of your 5 questions MUST be practical, logical programming algorithm/coding problems. For these 3 coding questions, you MUST set exactly 'type': 'Coding' and clearly denote expected Inputs and Outputs."
    
    llm = ChatOllama(model="llama3.2:1b", temperature=0.7)
    
    # Prompt the LLM to act as a modern FAANG interviewer
    prompt = f"""You are an elite FAANG Engineering Manager conducting a technical interview in 2026.
The candidate is interviewing for a '{role}' position and claims expertise in: {', '.join(pool)}.
The interview difficulty is: {difficulty.upper()}.

Using the latest internet trends and modern industry standards, formulate exactly 5 highly realistic, scenario-based interview questions. 
Do NOT ask basic definitions (e.g., "What is Python?"). Instead, ask complex, situational, or architectural questions that test deep understanding.{coding_instruction}
Scale the harshness and complexity to the {difficulty} level.

Return ONLY a valid JSON array of objects. EVERY object must possess:
"text": The question itself.
"type": A short categorization (e.g., "System Design", "Deep Dive: {pool[0]}", "Behavioral").

Example format:
[
  {{"text": "Your intricate situational question here...", "type": "Architecture"}},
  {{"text": "Write a function to...", "type": "Coding"}}
]

DO NOT output any markdown blocks, greetings, or conversational text. Return plain JSON.
"""
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        content = response.content.strip()
        
        # Aggressive JSON extraction using Regex in case LLM is conversational
        import re
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
            
        questions = json.loads(content)
        
        # Ensure we return at least 5 questions 
        if isinstance(questions, list) and len(questions) >= 5:
            # Let's shuffle them slightly to ensure variety if it outputs the same
            random.shuffle(questions)
            return questions[:5]
            
    except Exception as e:
        print(f"LLM Generation failed, using intelligent fallback. Error: {e}")
        
    # Intelligent Fallback mechanism in case the LLM fails to output valid JSON
    fallback_questions = []
    
    coding_templates = [
        "Given an array of integers, write a function to return the indices of the two numbers that add up to a specific target.",
        "Implement an algorithm to determine if a string has all unique characters.",
        "Write a method to replace all spaces in a string with '%20'.",
        "Implement a function to check if a linked list is a palindrome.",
        "Write a program to find the longest common prefix string amongst an array of strings.",
        "Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
        "Merge two sorted linked lists and return it as a new sorted list.",
        "Given an integer array nums, find the contiguous subarray which has the largest sum and return its sum.",
        "Climbing Stairs: You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?",
        "Write a function to reverse a string in-place."
    ]
    
    behavioral_templates = [
        "Describe your typical daily workflow when using {} for a standard project.",
        "What are the biggest advantages and disadvantages of using {} compared to its competitors?",
        "Can you explain a basic implementation of {} to a non-technical stakeholder?",
        "Where did you first learn {}, and how do you stay updated with its latest releases?",
        f"What is the most useful feature of {{}} that you use regularly as a {role}?",
        "Walk me through a specific time you had to optimize the performance of an application heavily relying on {}.",
        "What design patterns do you consider essential when bootstrapping a new modular architecture using {}?",
        "If a critical production service built with {} went down, what is your step-by-step debugging workflow?",
        "How does {} fit into your CI/CD pipeline, and what automated testing strategies do you employ for it?",
        f"As a {role}, describe a scenario where you intentionally accepted technical debt related to {{}}.",
        "Given your experience with {}, how would you architecture a globally distributed system that handles 10M concurrent users with zero downtime?",
        "Describe the most complex race condition or concurrency bug you've encountered in {}, and how you definitively solved it."
    ]
        
    # Pick exactly 3 coding problems
    selected_coding = random.sample(coding_templates, 3)
    for q in selected_coding:
        fallback_questions.append({
            "text": q,
            "type": "Coding"
        })
        
    # Pick exactly 2 behavioral/system problems
    selected_behavioral = random.sample(behavioral_templates, 2)
    for q in selected_behavioral:
        skill = random.choice(pool)
        fallback_questions.append({
            "text": q.format(skill),
            "type": f"Round [{difficulty.capitalize()}]: {skill} Deep Dive"
        })
        
    random.shuffle(fallback_questions)
    return fallback_questions


def mock_evaluate_response(question, response):
    words_list = [w for w in re.split(r'\s+', response) if len(w.strip()) > 0]
    words = len(words_list)

    if words < 5:
        return {
            "score": 0,
            "feedback": "Your answer was blank or far too short to be evaluated. In a real interview, always provide a detailed technical explanation.",
            "accuracy": 0, "clarity": 0, "confidence": 0,
            "toneAnalysis": "No significant audio or text detected."
        }

    hesitations = len(re.findall(r'(?i)\b(um|uh|like|you know|sort of|basically)\b', response))
    confidence_score = max(0, 95 - (hesitations * 15))
    tone_msg = "Tone was confident and steady." if hesitations == 0 else "Significant hesitations detected (e.g., 'um', 'uh'). Practice pausing silently instead."

    target_skill = ""
    target_match = re.search(r'(?i)knowledge of (.*?) in', question) or re.search(r'(?i)working with (.*?),', question) or re.search(r'(?i)strategy around (.*?)\?', question)
    if target_match and target_match.group(1):
        target_skill = re.sub(r'[^a-z0-9]', '', target_match.group(1).lower())

    is_relevant = False
    res_words = response.lower()
    if target_skill and target_skill in res_words:
        is_relevant = True

    tech_keywords = re.findall(r'(?i)\b(api|database|system|component|test|deploy|design|code|framework|architecture|problem|solution|function|class|variable|implement|scale|server|client|front|back|end|data)\b', response)
    keyword_hits = len(tech_keywords)
    if keyword_hits >= 2:
        is_relevant = True

    score = 0
    text_feedback = ""

    # Check if this includes Virtual Compiler Sandbox output
    passed_tests = len(re.findall(r'(?i)\bpass\b', response))
    failed_tests = len(re.findall(r'(?i)\bfail\b', response))
    
    if passed_tests > 0 or failed_tests > 0:
        is_relevant = True
        score = 50 + (passed_tests * 15) - (failed_tests * 10)
        score = min(100, max(10, score))
        if passed_tests > failed_tests:
            text_feedback = f"Great coding! You passed multiple test cases ({passed_tests} PASS). Your logic was sound."
        else:
            text_feedback = f"Your code failed several test cases ({failed_tests} FAIL). Consider edge cases and algorithmic bounds."
    elif not is_relevant:
        score = 25
        text_feedback = "Your response was completely off-topic or irrelevant. You did not directly address the core technical concept of the question. Make sure you stay on topic and reference the specific tools mentioned."
    else:
        score = 80 if words > 20 else max(40, words * 2)
        score += (keyword_hits * 2)
        score = min(100, score)
        if words > 20 and keyword_hits >= 2:
            text_feedback = "Excellent content! You effectively utilized technical terminology and provided a structured, accurate answer."
        elif words > 20:
            text_feedback = "Good length, but incorporate more specific technical keywords related to the task to demonstrate deeper expertise."
        else:
            text_feedback = "Your answer was relevant but a bit brief. Next time, provide more technical detail."

    return {
        "score": score, "feedback": text_feedback,
        "accuracy": min(100, score + 10) if is_relevant else 10,
        "clarity": min(100, score + 5),
        "confidence": confidence_score, "toneAnalysis": tone_msg
    }


@mock_interview_bp.route('/generate-questions', methods=['POST'])
def generate_questions():
    data = request.get_json()
    role = data.get('role')
    skills = data.get('skills', [])
    difficulty = data.get('difficulty', 'medium')
    
    if not role or not skills:
        return jsonify({"error": "Role and skills required"}), 400
        
    questions = mock_generate_questions(role, skills, difficulty)
    return jsonify({"questions": questions})

@mock_interview_bp.route('/evaluate-response', methods=['POST'])
def evaluate_answer():
    data = request.get_json()
    question = data.get('question', '')
    response_text = data.get('response', '')
    
    evaluation = mock_evaluate_response(question, response_text)
    return jsonify({"evaluation": evaluation})

@mock_interview_bp.route('/save-analytics', methods=['POST'])
def save_interview_analytics():
    return jsonify({"message": "Analytics saved successfully", "id": "mock_id_123"})

@mock_interview_bp.route('/career-roadmap', methods=['POST'])
def generate_career_roadmap():
    data = request.get_json()
    role = data.get('role')
    score = data.get('score')
    skills = data.get('skills', [])
    
    if not role or score is None:
        return jsonify({"error": "Role and score required"}), 400
        
    dominant_skill = skills[0] if skills else "Software Engineering"
    roadmap = []
    
    if score >= 75:
        roadmap = [
            {"title": "Aggressive Application Phase", "timeframe": "Month 1", "description": f"Your impressive score of {score}% indicates you are highly competitive for {role} positions right now. Prioritize outbound networking over passive studying.", "actionItems": [f"Tailor your resume explicitly targeting mid-level or advanced {role} roles.", f"Highlight your deep understanding of {dominant_skill} in your cover letters.", "Reach out to specific technical recruiters on LinkedIn with your portfolio."]},
            {"title": "Interview Mastery & Negotiation", "timeframe": "Month 2", "description": "As you land interviews, pivot towards mastering system architecture and behavioral STAR responses.", "actionItems": ["Practice advanced mock interviews focusing purely on edge-case scenarios.", "Research standard salary negotiations and compensation packages for your local area.", "Prepare questions to ask the interviewers about their tech stack."]},
            {"title": "Onboarding & Senior Pathway", "timeframe": "Month 3-6", "description": "Upon landing the job, establish yourself as a domain expert quickly.", "actionItems": [f"Take ownership of a project involving {dominant_skill} within your first 90 days.", "Schedule regular 1-on-1s with senior engineers to accelerate your learning.", "Start contributing to your company's internal documentation."]}
        ]
    elif score >= 40:
        roadmap = [
            {"title": "Core Foundational Review", "timeframe": "Month 1-2", "description": f"You have an acceptable base score of {score}%, but you need to eliminate vocal hesitations and expand your theoretical knowledge.", "actionItems": [f"Dedicate 10 hours a week to studying advanced concepts of {dominant_skill}.", "Start a blog or dev journal explaining the technical concepts you struggle with.", f"Take an additional online certification related to the {role} field."]},
            {"title": "Portfolio Project Expansion", "timeframe": "Month 3-4", "description": "Your vocabulary is good but lacks practical evidence. You need hands-on proof of skill.", "actionItems": ["Build a full-stack project from scratch and host it live.", f"Ensure the project heavily relies on {dominant_skill} to demonstrate competence.", "Contribute to one open-source repository."]},
            {"title": "Job Readiness & Mock Loop", "timeframe": "Month 5-6", "description": "Begin entering the job market while continuing to refine your interview skills.", "actionItems": ["Take this Mock Interview again aiming for a score > 75%.", f"Start applying to Junior or Entry-Level {role} positions.", "Optimize your LinkedIn profile with your new portfolio links."]}
        ]
    else:
        roadmap = [
            {"title": "Intensive Skill Bootstrapping", "timeframe": "Month 1-3", "description": f"Your score of {score}% indicates a critical gap in fundamental knowledge. Before scheduling interviews, you must build a solid baseline.", "actionItems": [f"Enroll in a structured Bootcamp or rigorous course specifically for {dominant_skill}.", "Dedicate time strictly to building small, functional tutorial projects.", "Focus on learning the vocabulary; study glossaries of common tech terms."]},
            {"title": "Intermediate Concept Application", "timeframe": "Month 4-6", "description": "Start transitioning from tutorial learning to independent building.", "actionItems": ["Build two separate solo projects without following a guide.", "Implement basic testing and deployment pipelines.", "Join a local tech meetup or Discord community for code-reviews."]},
            {"title": "Resume Building & Market Entry", "timeframe": "Month 6+", "description": "Do not rush into interviews before you have a proper portfolio.", "actionItems": ["Draft your first tech resume highlighting your new solo projects.", "Retake this Mock AI simulator and track your score improvement.", "Start applying for internships or strictly junior roles."]}
        ]
        
    return jsonify({"roadmap": roadmap})

@mock_interview_bp.route('/internship-match', methods=['POST'])
def evaluate_internships():
    data = request.get_json()
    score = data.get('score')
    skills = data.get('skills')
    if score is None or not skills:
        return jsonify({"error": "Score and skills required"}), 400
        
    skill_str = " ".join(skills).lower()
    possible_roles = [
        {"title": "AI / ML Engineering Intern", "keywords": ['python', 'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'data', 'model', 'ai', 'nlp']},
        {"title": "Frontend Engineering Intern", "keywords": ['react', 'javascript', 'html', 'css', 'ui', 'frontend', 'next']},
        {"title": "Backend Systems Intern", "keywords": ['node', 'python', 'java', 'sql', 'database', 'api', 'backend', 'express']},
        {"title": "Data Analyst Intern", "keywords": ['python', 'sql', 'data', 'excel', 'machine learning', 'analytics', 'pandas']},
        {"title": "DevOps & Cloud Intern", "keywords": ['docker', 'aws', 'cloud', 'linux', 'ci/cd', 'deploy', 'kubernetes']},
        {"title": "Full Stack Developer Intern", "keywords": ['react', 'node', 'javascript', 'database', 'api', 'fullstack', 'next']},
        {"title": "Product Management Intern", "keywords": ['leadership', 'agile', 'scrum', 'business', 'management', 'product']}
    ]
    
    matches = []
    for role_data in possible_roles:
        keyword_matches = sum(1 for kw in role_data["keywords"] if kw in skill_str)
        skill_compatibility = min(40, keyword_matches * 15)
        performance_compatibility = (score / 100) * 60
        final_percentage = round(skill_compatibility + performance_compatibility + (random.random() * 5))
        final_percentage = min(99, max(15, final_percentage))
        
        if keyword_matches == 0 and final_percentage > 50:
            final_percentage = round(final_percentage * 0.6)
            
        matches.append({"role": role_data["title"], "percentage": final_percentage})
        
    matches.sort(key=lambda x: x["percentage"], reverse=True)
    return jsonify({"internships": matches[:4]})

@mock_interview_bp.route('/run-code', methods=['POST'])
def virtual_sandbox():
    data = request.json
    question = data.get('question', '')
    code = data.get('code', '')

    if not code.strip():
        return jsonify({"output": "[Error] Terminal Error: No code provided to the compiler."}), 400

    from langchain_ollama import ChatOllama
    from langchain_core.messages import SystemMessage
    llm = ChatOllama(model="llama3.2:1b", temperature=0.1)

    prompt = f"""You are a strict, ultra-fast Virtual Code Compiler and Test-Case Runner.
Do not talk to the user. Do not explain anything. 

PROBLEM ASSIGNED TO USER: 
{question}

USER'S CODE TO RUN:
{code}

YOUR ROLE: 
Silently figure out what the optimal solution is. Then, create 3 hidden test cases with diverse inputs in your sandbox. 
Mentally 'execute' the user's code against those 3 test cases. 
Format your output exactly like a cold, emotionless Terminal console. 
If there are syntax errors or missing imports or obvious logical infinite loops, fail them immediately with a traceback-style error sequence.

Output strictly in this format ONLY:

[Compiler: OK. Running Test Cases...]

Test Case 1 (Standard): PASS / FAIL [Reason if fail]
Test Case 2 (Edge Case): PASS / FAIL [Reason if fail]
Test Case 3 (Large Input): PASS / FAIL [Reason if fail]

Terminal execution finished.
"""
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        return jsonify({"output": response.content.strip()})
    except Exception as e:
        return jsonify({"output": f"[Virtual Sandbox OS Error] Connection to Code Runner Interrupted: {str(e)}"}), 500

@mock_interview_bp.route('/profile/extract', methods=['POST'])
def auto_fill_profile():
    stream = get_resume_stream_from_req(request)
    if not stream:
        return jsonify({"error": "No resume found. Please provide session_id or file."}), 400

    # ── STEP 1: Full PDF Text Extraction ───────────────────────────────
    text = extract_text_from_pdf(stream)
    print(f"[Profile Extract] Extracted {len(text)} chars from PDF.")

    if not text or len(text) < 30:
        return jsonify({"error": "Could not extract text from the PDF."}), 400

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text_lower = text.lower()

    # ── STEP 2: Email ───────────────────────────────────────────────────
    email_match = re.search(r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}', text)
    extracted_email = email_match.group(0) if email_match else ""

    # ── STEP 3: Full Name (3 strategies) ────────────────────────────────
    extracted_name = ""
    # Strategy A: first short capitalised alphabetic line (top 20 lines)
    for line in lines[:20]:
        words = line.split()
        if 2 <= len(words) <= 5 and re.match(r'^[A-Za-z .\'\-]+$', line) and len(line) > 5:
            # Reject lines that look like headers (all caps single word, dates, URLs)
            if not re.search(r'\d', line) and 'http' not in line.lower():
                extracted_name = line.strip()
                break
    # Strategy B: explicit label like "Name: John Doe"
    if not extracted_name:
        nm = re.search(r'(?:name|candidate)\s*[:\-]\s*([A-Za-z .\'\-]{4,50})', text, re.IGNORECASE)
        if nm:
            extracted_name = nm.group(1).strip()
    # Strategy C: derive from email local part (e.g. john.doe@...)
    if not extracted_name and extracted_email:
        local = extracted_email.split('@')[0]
        parts = re.split(r'[._\-]', local)
        if len(parts) >= 2:
            extracted_name = ' '.join(p.capitalize() for p in parts if p.isalpha())

    # ── STEP 4: Phone ────────────────────────────────────────────────────
    phone_match = re.search(r'(?:\+?\d[\s\-]?)?\(?\d{3,5}\)?[\s\-]?\d{3,5}[\s\-]?\d{3,5}', text)
    extracted_phone = phone_match.group(0).strip() if phone_match else ""

    # ── STEP 5: Graduation years ─────────────────────────────────────────
    # Find all years, prefer those near education/degree keywords
    all_year_matches = list(re.finditer(r'\b(19[9][0-9]|20[0-3][0-9])\b', text))
    edu_keywords = ['b.tech','b.e','m.tech','mba','bca','mca','bachelor','master','degree',
                    'university','college','institute','graduation','cgpa','gpa']
    edu_positions = [m.start() for kw in edu_keywords for m in re.finditer(kw, text_lower)]

    def near_edu(pos):
        return any(abs(pos - ep) < 300 for ep in edu_positions)

    edu_years   = sorted(set(int(m.group()) for m in all_year_matches if near_edu(m.start())))
    all_years   = sorted(set(int(m.group()) for m in all_year_matches))
    year_pool   = edu_years if edu_years else all_years

    grad_year  = str(year_pool[0])  if year_pool else ""
    pg_year     = str(year_pool[-1]) if len(year_pool) > 1 and year_pool[-1] != (year_pool[0] if year_pool else 0) else ""

    # ── STEP 6: Country ──────────────────────────────────────────────────
    country = ""
    for c in ["USA","United States","UK","United Kingdom","Canada","Australia",
              "Germany","France","Singapore","UAE","Dubai","Nepal",
              "Bangladesh","Sri Lanka","India"]:
        if c.lower() in text_lower:
            country = c
            break
    if not country:
        country = "India"

    # ── STEP 7: University ───────────────────────────────────────────────
    university = ""
    uni_pats = [
        r'([A-Z][a-zA-Z ]+ (?:University|College|Institute of Technology|Institute|Academy|Polytechnic|School of))',
        r'((?:IIT|NIT|IIIT|BITS|VIT|SRM|Anna|Amrita|Manipal|Lovely Professional)[A-Za-z ,\-]{2,50})',
        r'(?:from|at|pursuing|studied at)\s+([A-Z][a-zA-Z ]{5,50})'
    ]
    for pat in uni_pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            university = m.group(0).strip()[:80]
            break

    # ── STEP 8: Degree ───────────────────────────────────────────────────
    degree = ""
    deg_pats = [
        r'B\.?\s*Tech\.?[^\n,]{0,45}', r'B\.?\s*E\.?[^\n,]{0,30}',
        r'M\.?\s*Tech\.?[^\n,]{0,45}', r'MBA[^\n,]{0,30}',
        r'BCA[^\n,]{0,30}', r'MCA[^\n,]{0,30}',
        r'B\.?\s*Sc\.?[^\n,]{0,40}', r'M\.?\s*Sc\.?[^\n,]{0,40}',
        r'Bachelor[^\n,]{0,55}', r'Master[^\n,]{0,55}',
        r'B\.?\s*Com[^\n,]{0,30}', r'Ph\.?\s*D\.?[^\n,]{0,30}'
    ]
    for pat in deg_pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            degree = m.group(0).strip()[:65]
            break

    # ── STEP 9: Skills (full SKILLS_DB scan on entire text) ───────────────
    found_skills = extract_skills(text)  # scans entire text
    skills_str = ", ".join(found_skills[:25]) if found_skills else ""
    print(f"[Profile Extract] Skills found: {len(found_skills)} | Name: '{extracted_name}' | Uni: '{university}'")

    # ── STEP 10: Certifications ───────────────────────────────────────────
    cert_raw = re.findall(
        r'(?:AWS|Azure|GCP|Google Cloud|Oracle|Cisco|CompTIA|CEH|PMP|Scrum Master|ISTQB|Certified[^\n]{0,30})[^\n,]{0,60}',
        text, re.IGNORECASE
    )
    certifications = ", ".join(dict.fromkeys(c.strip()[:70] for c in cert_raw[:6]))

    final_profile = {
        "fullName":       extracted_name,
        "email":          extracted_email,
        "phone":          extracted_phone,
        "dob":            "",
        "fathersName":    "",
        "bloodGroup":     "",
        "country":        country,
        "skills":         skills_str,
        "universityName": university,
        "id":             "",
        "courseOfDegree": degree,
        "gradYear":       grad_year,
        "postGradYear":   pg_year,
        "certifications": certifications
    }

    return jsonify({"profile": final_profile})



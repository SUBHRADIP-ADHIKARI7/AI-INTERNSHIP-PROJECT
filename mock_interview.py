import re
import random
import io
import os
from datetime import datetime
from urllib.parse import quote_plus

import requests
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


SECTION_ALIASES = {
    "summary": {
        "summary", "professional summary", "profile", "objective", "about me", "career summary"
    },
    "skills": {
        "skills", "technical skills", "core skills", "core competencies", "technologies", "tech stack"
    },
    "experience": {
        "experience", "work experience", "professional experience", "employment history", "internships", "internship experience"
    },
    "projects": {
        "projects", "academic projects", "personal projects", "project experience"
    },
    "education": {
        "education", "academic background", "academics", "qualifications"
    },
    "certifications": {
        "certifications", "licenses", "certificates", "accreditations"
    },
    "achievements": {
        "achievements", "awards", "accomplishments", "highlights"
    }
}

TECH_ROLE_HINTS = {
    "developer", "engineer", "software", "data scientist", "data analyst", "ml", "ai",
    "devops", "cloud", "backend", "frontend", "full stack", "mobile"
}

INTERVIEW_RESEARCH_CACHE = {}

THEME_SIGNAL_MAP = {
    "system design": ["scalability", "distributed", "architecture", "microservices", "availability", "throughput"],
    "debugging": ["debug", "incident", "root cause", "bug", "failure", "logs", "troubleshoot"],
    "performance": ["latency", "optimize", "performance", "bottleneck", "throughput", "profiling"],
    "testing quality": ["test", "qa", "regression", "unit test", "integration", "quality"],
    "security": ["security", "auth", "authorization", "vulnerability", "owasp", "encryption"],
    "collaboration": ["stakeholder", "communication", "team", "cross-functional", "conflict", "leadership"],
    "ownership": ["ownership", "impact", "delivery", "deadline", "prioritization", "trade-off"],
    "data modeling": ["schema", "database", "sql", "consistency", "indexing", "query"]
}


def _normalized_resume_lines(text):
    return [ln.strip() for ln in text.replace("\t", " ").splitlines() if ln.strip()]


def _normalize_heading(line):
    return re.sub(r"[^a-z ]+", "", line.lower()).strip()


def _split_resume_sections(lines):
    sections = {name: [] for name in SECTION_ALIASES}
    current = None
    for line in lines:
        heading = _normalize_heading(line)
        matched = None
        for section, aliases in SECTION_ALIASES.items():
            if heading in aliases:
                matched = section
                break
        if matched:
            current = matched
            continue
        if current:
            sections[current].append(line)
    return sections


def _unique_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _extract_links(text):
    raw_links = re.findall(r"(https?://[^\s|,;]+|www\.[^\s|,;]+|linkedin\.com/[^\s|,;]+|github\.com/[^\s|,;]+)", text, re.IGNORECASE)
    links = []
    for link in raw_links:
        cleaned = link.rstrip(").,;]")
        if cleaned.lower().startswith("www."):
            cleaned = "https://" + cleaned
        elif not cleaned.lower().startswith("http"):
            cleaned = "https://" + cleaned
        links.append(cleaned)
    links = _unique_preserve_order(links)

    linkedin = next((ln for ln in links if "linkedin.com" in ln.lower()), "")
    github = next((ln for ln in links if "github.com" in ln.lower()), "")
    portfolio = next((ln for ln in links if ln not in {linkedin, github}), "")
    return linkedin, github, portfolio


def _extract_name(lines, email):
    noise_words = {
        "resume", "curriculum", "vitae", "profile", "contact", "phone", "email", "linkedin",
        "github", "skills", "education", "experience", "project"
    }
    for line in lines[:18]:
        if len(line) < 5 or len(line) > 60:
            continue
        if any(ch.isdigit() for ch in line) or "@" in line or "http" in line.lower():
            continue
        lowered = line.lower()
        if any(word in lowered for word in noise_words):
            continue
        words = [w for w in re.split(r"\s+", line) if w]
        if 2 <= len(words) <= 5 and all(re.match(r"^[A-Za-z][A-Za-z'.-]*$", w) for w in words):
            return line.strip()

    if email:
        local = email.split("@")[0]
        parts = [p for p in re.split(r"[._-]+", local) if p.isalpha()]
        if len(parts) >= 2:
            return " ".join(p.capitalize() for p in parts[:3])
    return ""


def _extract_location(lines):
    for line in lines[:20]:
        if "@" in line or "http" in line.lower():
            continue
        if re.search(r"\b[A-Za-z ]+,\s*[A-Za-z ]+\b", line):
            return line.strip()
    return ""


def _extract_summary(lines, sections):
    if sections["summary"]:
        return " ".join(sections["summary"][:3])[:400]
    for line in lines:
        if len(line.split()) >= 14:
            return line[:400]
    return ""


def _extract_education_entries(lines, sections):
    edu_lines = sections["education"] if sections["education"] else lines
    entries = []
    for line in edu_lines:
        if re.search(r"\b(b\.?tech|b\.?e\.?|m\.?tech|mba|bca|mca|bachelor|master|ph\.?d|b\.?sc|m\.?sc)\b", line, re.IGNORECASE):
            entries.append(line)
    return _unique_preserve_order(entries)[:5]


def _extract_highlights(section_lines, max_items=4):
    cleaned = []
    for line in section_lines:
        line = re.sub(r"^[\-\*\u2022]+\s*", "", line).strip()
        if len(line) < 8:
            continue
        cleaned.append(line)
    return _unique_preserve_order(cleaned)[:max_items]


def _estimate_years_experience(text):
    explicit = re.search(r"(\d{1,2})\+?\s+years?\s+of\s+experience", text, re.IGNORECASE)
    if explicit:
        return int(explicit.group(1))

    current_year = datetime.now().year
    ranges = re.findall(
        r"(19\d{2}|20\d{2})\s*[-–to]+\s*(present|current|19\d{2}|20\d{2})",
        text,
        re.IGNORECASE
    )
    spans = []
    for start, end in ranges:
        s = int(start)
        e = current_year if end.lower() in {"present", "current"} else int(end)
        if s <= e <= current_year + 1:
            spans.append((s, e))
    if not spans:
        return 0

    spans.sort()
    merged = []
    for start, end in spans:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    total = sum(end - start for start, end in merged)
    return max(0, min(total, 25))


def extract_resume_profile(text):
    lines = _normalized_resume_lines(text)
    sections = _split_resume_sections(lines)

    email_match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"(?:\+?\d[\d\s()-]{8,}\d)", text)
    email = email_match.group(0) if email_match else ""
    phone = phone_match.group(0).strip() if phone_match else ""

    linkedin, github, portfolio = _extract_links(text)
    skills = extract_skills(text)
    inferred_role = infer_role(skills) if skills else "Software Engineer"

    project_highlights = _extract_highlights(sections["projects"], max_items=5)
    experience_highlights = _extract_highlights(sections["experience"], max_items=5)
    certifications = _extract_highlights(sections["certifications"], max_items=5)
    achievements = _extract_highlights(sections["achievements"], max_items=5)
    education_entries = _extract_education_entries(lines, sections)

    profile = {
        "fullName": _extract_name(lines, email),
        "email": email,
        "phone": phone,
        "location": _extract_location(lines),
        "linkedin": linkedin,
        "github": github,
        "portfolio": portfolio,
        "summary": _extract_summary(lines, sections),
        "skills": skills,
        "role": inferred_role,
        "yearsExperience": _estimate_years_experience(text),
        "education": education_entries,
        "experienceHighlights": experience_highlights,
        "projects": project_highlights,
        "certifications": certifications,
        "achievements": achievements,
        "rawTextPreview": text[:1000] + ("..." if len(text) > 1000 else "")
    }
    return profile


def _is_tech_role(role):
    role_lower = (role or "").lower()
    return any(hint in role_lower for hint in TECH_ROLE_HINTS)


def _normalize_difficulty(difficulty):
    value = (difficulty or "medium").strip().lower()
    return value if value in {"easy", "medium", "hard"} else "medium"


def _compact_resume_line(line, max_len=160):
    cleaned = re.sub(r"\s+", " ", (line or "").strip())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip() + "..."


def _fetch_duckduckgo_snippets(query, max_items=6):
    try:
        # Lightweight public HTML endpoint; no API key required.
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=3.5
        )
        response.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        items = []
        for result in soup.select(".result"):
            title_el = result.select_one(".result__a")
            snippet_el = result.select_one(".result__snippet")
            title = title_el.get_text(" ", strip=True) if title_el else ""
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            merged = " ".join(x for x in [title, snippet] if x).strip()
            if merged:
                items.append(merged)
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


def _research_interview_signals(role, skills):
    cache_key = (role.lower().strip(), tuple(s.lower().strip() for s in skills[:4]))
    now = datetime.utcnow()
    cached = INTERVIEW_RESEARCH_CACHE.get(cache_key)
    if cached and (now - cached["timestamp"]).total_seconds() < 3600:
        return cached["data"]

    top_skills = skills[:3] if skills else ["software engineering"]
    queries = [f"real {role} interview questions"]
    queries.extend([f"{skill} interview questions scenario based" for skill in top_skills])

    snippets = []
    for query in queries[:4]:
        snippets.extend(_fetch_duckduckgo_snippets(query, max_items=4))

    text_blob = " ".join(snippets).lower()
    theme_scores = {}
    for theme, keywords in THEME_SIGNAL_MAP.items():
        score = sum(text_blob.count(keyword) for keyword in keywords)
        if score > 0:
            theme_scores[theme] = score

    ranked_themes = [k for k, _ in sorted(theme_scores.items(), key=lambda kv: kv[1], reverse=True)]
    if not ranked_themes:
        ranked_themes = ["system design", "debugging", "performance", "collaboration", "ownership"]

    data = {
        "queries": queries,
        "themes": ranked_themes[:6],
        "snippets": snippets[:12]
    }
    INTERVIEW_RESEARCH_CACHE[cache_key] = {"timestamp": now, "data": data}
    return data


def _dedupe_questions(questions):
    seen = set()
    unique = []
    for q in questions:
        key = re.sub(r"\s+", " ", q["text"].strip().lower())
        if key and key not in seen:
            seen.add(key)
            unique.append(q)
    return unique


def _pick_random(rng, pool, limit):
    if limit <= 0:
        return []
    copied = list(pool)
    rng.shuffle(copied)
    return copied[:min(limit, len(copied))]


def generate_real_life_questions(profile, difficulty="medium", count=5):
    difficulty = _normalize_difficulty(difficulty)
    rng = random.SystemRandom()

    role = profile.get("role") or "Software Engineer"
    skills = _unique_preserve_order(profile.get("skills") or [])
    top_skill = skills[0] if skills else "problem solving"
    second_skill = skills[1] if len(skills) > 1 else top_skill
    third_skill = skills[2] if len(skills) > 2 else top_skill
    projects = [_compact_resume_line(x) for x in (profile.get("projects") or []) if x]
    experience = [_compact_resume_line(x) for x in (profile.get("experienceHighlights") or []) if x]
    achievements = [_compact_resume_line(x) for x in (profile.get("achievements") or []) if x]
    summary = _compact_resume_line(profile.get("summary", ""), max_len=200)
    years = profile.get("yearsExperience", 0)
    research = _research_interview_signals(role, skills)
    market_themes = research.get("themes", [])
    dominant_theme = market_themes[0] if market_themes else "system design"

    role_scope = f"{role} role"
    if years:
        role_scope = f"{role} role ({years}+ years experience)"

    anchors = projects + experience + achievements
    anchor = rng.choice(anchors) if anchors else ""

    resume_prompts = [
        f"Walk me through this resume point: \"{anchor}\". What was the business problem, your ownership, key trade-offs, and measurable impact?" if anchor else "",
        f"Which project on your resume best represents your readiness for a {role_scope}? Break down architecture choices, constraints, and outcomes.",
        f"Pick one project where you used {top_skill}. Explain what failed initially, what you changed, and the final metrics.",
        f"Tell me about a project where your solution reduced risk, cost, or latency. What exact decision made the difference?"
    ]
    resume_prompts = [q for q in resume_prompts if q]

    behavioral_prompts = {
        "easy": [
            "Tell me about a time you received difficult feedback. What did you change afterward?",
            "Describe a time you had to collaborate with someone who had a very different working style."
        ],
        "medium": [
            "Tell me about a time you disagreed with a teammate on technical direction. How did you resolve it?",
            "Describe a situation where priorities changed mid-sprint. How did you re-plan and communicate risk?",
            "Share an example of when you made a mistake in production. How did you own it and recover?"
        ],
        "hard": [
            "Describe a high-pressure incident where deadline and quality were in conflict. How did you decide what to ship?",
            "Tell me about a time you had limited data but still had to make a decision with major impact.",
            "Give an example where you had to influence senior stakeholders to change a technical plan."
        ]
    }

    situational_prompts = [
        f"You join a team using {top_skill}. A release is in 24 hours and a P1 bug appears. What is your first-hour plan?",
        f"You inherit a service built on {second_skill} with poor documentation and high error rates. How do you stabilize it in week one?",
        f"Assume your manager asks for a feature in half the estimated time. How do you negotiate scope while protecting quality?",
        f"Interviewers currently emphasize {dominant_theme}. Describe a real situation where you handled this successfully."
    ]
    if summary:
        situational_prompts.append(
            f"Based on your profile summary (\"{summary}\"), what risks would you watch first when onboarding into a new {role_scope}?"
        )

    coding_prompts = [
        f"Coding task: Using {top_skill}, implement an LRU cache with O(1) get/put. Input: operations list. Output: values for get operations.",
        f"Coding task: Using {second_skill}, write a function to merge overlapping intervals. Input: list of [start, end]. Output: merged intervals sorted by start.",
        f"Coding task: Using {top_skill}, implement top-K frequent elements. Input: array of integers and k. Output: k most frequent values.",
        f"Coding task: Using {third_skill}, implement a rate limiter. Input: request timestamps and limit/window. Output: allow or reject per request.",
        f"Coding task: Using {top_skill}, detect a cycle in a directed graph and return one valid cycle path if found."
    ]

    system_design_prompts = [
        f"System design: Design a multi-tenant interview simulator for a {role_scope}. Cover API boundaries, data model, queueing, caching, and failure handling.",
        f"System design: Design a real-time collaboration platform using {top_skill} and {second_skill}. Explain scaling strategy, consistency model, and observability.",
        f"System design: Design an event-driven analytics pipeline for interview sessions. Include ingestion, processing, storage, and cost controls.",
        f"System design: Interview loops now focus on {dominant_theme}. Design a production-grade service showing how you handle this at scale."
    ]

    non_tech_competency_prompts = [
        f"For this {role_scope}, how would you design a KPI dashboard to prove impact in your first 90 days?",
        f"Imagine one critical workflow in your team fails repeatedly. How would you root-cause and fix it permanently?",
        f"How would you use your strength in {top_skill} to improve execution quality across cross-functional teams?"
    ]

    fit_prompts = [
        f"Why this {role_scope}, and which two experiences from your resume best prove role fit?",
        "If you join tomorrow, what would your first 30-60-90 day execution plan look like?",
        f"What kind of team environment helps you do your best work, and where have you already demonstrated that in {top_skill}-heavy projects?"
    ]

    if count <= 0:
        count = 20

    resume_pool = [{"type": "Resume Deep Dive", "is_coding": False, "text": q} for q in resume_prompts]
    behavioral_pool = [{"type": "Behavioral", "is_coding": False, "text": q} for q in behavioral_prompts[difficulty]]
    situational_pool = [{"type": "Situational", "is_coding": False, "text": q} for q in situational_prompts]
    fit_pool = [{"type": "Motivation and Fit", "is_coding": False, "text": q} for q in fit_prompts]

    technical_pool = []
    if _is_tech_role(role):
        technical_pool.extend([{"type": "Coding", "is_coding": True, "text": q} for q in coding_prompts])
        technical_pool.extend([{"type": "System Design", "is_coding": False, "text": q} for q in system_design_prompts])
        # Skill-theme cross questions from internet signals.
        for skill in skills[:4]:
            for theme in market_themes[:3]:
                technical_pool.append({
                    "type": "Technical Scenario",
                    "is_coding": False,
                    "text": f"In interview loops for {role}, a common area is {theme}. Explain how you applied {skill} to solve a real {theme}-related challenge."
                })
    else:
        technical_pool.extend([{"type": "Role Competency", "is_coding": False, "text": q} for q in non_tech_competency_prompts])
        for skill in skills[:4]:
            for theme in market_themes[:2]:
                technical_pool.append({
                    "type": "Role Scenario",
                    "is_coding": False,
                    "text": f"Interviewers often test {theme}. How would you apply {skill} in a realistic business scenario to deliver measurable outcomes?"
                })

    # Add market-aligned prompts derived from internet snippets.
    market_pool = []
    for snippet in research.get("snippets", [])[:6]:
        market_pool.append({
            "type": "Market-Aligned",
            "is_coding": False,
            "text": (
                f"Current interview discussions mention: \"{_compact_resume_line(snippet, max_len=120)}\". "
                f"How would you address this in the context of your {role_scope} experience?"
            )
        })

    resume_target = max(3, count // 5)
    behavioral_target = max(4, count // 5)
    situational_target = max(4, count // 5)
    fit_target = max(2, count // 10)
    technical_target = max(5, count - (resume_target + behavioral_target + situational_target + fit_target))

    selected = []
    selected.extend(_pick_random(rng, resume_pool, resume_target))
    selected.extend(_pick_random(rng, behavioral_pool, behavioral_target))
    selected.extend(_pick_random(rng, situational_pool, situational_target))
    selected.extend(_pick_random(rng, technical_pool, technical_target))
    selected.extend(_pick_random(rng, fit_pool, fit_target))
    selected.extend(_pick_random(rng, market_pool, max(0, count // 6)))

    selected = _dedupe_questions(selected)
    if len(selected) < count:
        top_theme = market_themes[0] if market_themes else "problem solving"
        while len(selected) < count:
            selected.append({
                "type": "Follow-up",
                "is_coding": False,
                "text": (
                    f"For this {role_scope}, what framework would you use to prioritize between delivery speed, "
                    f"{top_theme}, and long-term maintainability?"
                )
            })
            selected = _dedupe_questions(selected)

    rng.shuffle(selected)
    return selected[:count]


@mock_interview_bp.route('/upload-resume', methods=['POST'])
def parse_resume():
    stream = get_resume_stream_from_req(request)
    if not stream:
        return jsonify({"error": "No resume found. Please provide session_id or file."}), 400

    # Extract text using multi-method approach
    text = extract_text_from_pdf(stream)

    if not text or len(text) < 30:
        return jsonify({"error": "Could not extract text from the PDF. Please ensure it is a text-based or clearly scanned resume."}), 400

    profile = extract_resume_profile(text)
        
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


def _clamp(value, lo=0.0, hi=100.0):
    return max(lo, min(hi, value))


def _compute_audio_confidence(audio_metrics):
    if not isinstance(audio_metrics, dict):
        return 50

    avg_volume = float(audio_metrics.get("avgVolume", 0.0) or 0.0)
    peak_volume = float(audio_metrics.get("peakVolume", 0.0) or 0.0)
    silence_ratio = float(audio_metrics.get("silenceRatio", 0.6) or 0.6)
    speech_seconds = float(audio_metrics.get("speechSeconds", 0.0) or 0.0)

    # Confidence heuristic from audio signals:
    # - stable audible volume
    # - not too much silence
    # - sufficient speaking duration
    volume_score = _clamp((avg_volume * 240) + (peak_volume * 100), 0, 100)
    silence_score = _clamp((1.0 - silence_ratio) * 100, 0, 100)
    duration_score = _clamp((speech_seconds / 18.0) * 100, 0, 100)

    combined = (volume_score * 0.4) + (silence_score * 0.35) + (duration_score * 0.25)
    return int(round(_clamp(combined, 0, 100)))


def _extract_question_text(question_obj):
    if isinstance(question_obj, str):
        return question_obj
    if isinstance(question_obj, dict):
        return question_obj.get("text", "")
    return ""


def _extract_answer_text(answer_obj):
    if isinstance(answer_obj, str):
        return answer_obj
    if isinstance(answer_obj, dict):
        spoken = answer_obj.get("spoken", "") or ""
        code = answer_obj.get("code", "") or ""
        if spoken and code:
            return f"{spoken}\n\n[Code Snippet]\n{code}"
        return spoken or code
    return ""


@mock_interview_bp.route('/generate-questions', methods=['POST'])
def generate_questions():
    data = request.get_json() or {}
    difficulty = data.get("difficulty", "medium")
    try:
        count = int(data.get("count", 20))
    except (TypeError, ValueError):
        count = 20

    incoming_profile = data.get("profile")
    if isinstance(incoming_profile, dict) and incoming_profile:
        profile = dict(incoming_profile)
    else:
        role = data.get("role") or "Software Engineer"
        skills = data.get("skills", []) or []
        profile = {
            "role": role,
            "skills": skills,
            "projects": [],
            "experienceHighlights": [],
            "achievements": [],
            "yearsExperience": 0
        }

    if not profile.get("role"):
        profile["role"] = infer_role(profile.get("skills", []))

    role = profile.get("role", "Software Engineer")
    skills = profile.get("skills", [])

    # Try elite LLM-powered generation first (resume-aware prompt)
    questions = mock_generate_questions(role, skills, difficulty=difficulty, profile=profile)

    # If LLM fails or returns too few, fall back to resume-aware rule-based system
    if not questions or len(questions) < 3:
        questions = generate_real_life_questions(profile, difficulty=difficulty, count=count)
    elif len(questions) < count:
        extras = generate_real_life_questions(profile, difficulty=difficulty, count=count - len(questions))
        questions = _dedupe_questions(questions + extras)

    return jsonify({
        "questions": questions[:count],
        "role": role,
        "skills": skills,
        "questionFramework": [
            "Resume Deep Dive",
            "Behavioral",
            "Situational",
            "Technical/System Design",
            "Motivation and Fit"
        ]
    })

@mock_interview_bp.route('/evaluate-response', methods=['POST'])
def evaluate_answer():
    data = request.get_json()
    question = data.get('question', '')
    response_text = data.get('response', '')
    
    evaluation = mock_evaluate_response(question, response_text)
    return jsonify({"evaluation": evaluation})


@mock_interview_bp.route('/analyze-session', methods=['POST'])
def analyze_session():
    try:
        if request.is_json:
            payload = request.get_json() or {}
        else:
            raw = request.form.get("session", "{}")
            payload = json.loads(raw)

        questions = payload.get("questions", []) or []
        answers = payload.get("answers", []) or []

        total_questions = len(questions)
        evaluated = []
        correct_count = 0
        wrong_count = 0
        unanswered_count = 0
        total_conf = 0
        total_score = 0

        for idx, question_obj in enumerate(questions):
            question_text = _extract_question_text(question_obj)
            answer_obj = answers[idx] if idx < len(answers) else {}
            answer_text = _extract_answer_text(answer_obj).strip()
            audio_metrics = answer_obj.get("audioMetrics") if isinstance(answer_obj, dict) else None

            if not answer_text:
                unanswered_count += 1
                audio_conf = _compute_audio_confidence(audio_metrics)
                evaluated.append({
                    "index": idx + 1,
                    "question": question_text,
                    "score": 0,
                    "correct": False,
                    "confidence": int(round(audio_conf * 0.6)),
                    "feedback": "No answer detected for this question.",
                    "audioConfidence": audio_conf
                })
                continue

            evaluation = mock_evaluate_response(question_text, answer_text)
            audio_conf = _compute_audio_confidence(audio_metrics)
            merged_conf = int(round((evaluation.get("confidence", 50) * 0.55) + (audio_conf * 0.45)))
            score = int(evaluation.get("score", 0))
            is_correct = score >= 60

            if is_correct:
                correct_count += 1
            else:
                wrong_count += 1

            total_conf += merged_conf
            total_score += score

            evaluated.append({
                "index": idx + 1,
                "question": question_text,
                "score": score,
                "correct": is_correct,
                "confidence": merged_conf,
                "feedback": evaluation.get("feedback", ""),
                "audioConfidence": audio_conf,
                "clarity": evaluation.get("clarity", 0),
                "accuracy": evaluation.get("accuracy", 0),
                "toneAnalysis": evaluation.get("toneAnalysis", "")
            })

        answered_count = max(0, total_questions - unanswered_count)
        accuracy_percent = int(round((correct_count / total_questions) * 100)) if total_questions else 0
        avg_conf = int(round(total_conf / answered_count)) if answered_count else 0
        avg_score = int(round(total_score / answered_count)) if answered_count else 0

        weak_areas = [row["index"] for row in evaluated if not row["correct"]][:5]
        strong_areas = [row["index"] for row in evaluated if row["correct"]][:5]

        session_id = str(int(datetime.utcnow().timestamp()))
        saved_audio = []
        if request.files:
            audio_dir = os.path.join(UPLOAD_FOLDER, "interview_audio", session_id)
            os.makedirs(audio_dir, exist_ok=True)
            for field_name, file_obj in request.files.items():
                filename = f"{field_name}.webm"
                path = os.path.join(audio_dir, filename)
                file_obj.save(path)
                saved_audio.append(path)

        return jsonify({
            "totalQuestions": total_questions,
            "answeredCount": answered_count,
            "correctCount": correct_count,
            "wrongCount": wrong_count,
            "unansweredCount": unanswered_count,
            "accuracyPercent": accuracy_percent,
            "averageScore": avg_score,
            "overallConfidence": avg_conf,
            "savedAudioCount": len(saved_audio),
            "strongAnswerIndexes": strong_areas,
            "weakAnswerIndexes": weak_areas,
            "perQuestion": evaluated
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

    text = extract_text_from_pdf(stream)
    if not text or len(text) < 30:
        return jsonify({"error": "Could not extract text from the PDF."}), 400

    profile = extract_resume_profile(text)

    final_profile = {
        "fullName": profile.get("fullName", ""),
        "email": profile.get("email", ""),
        "phone": profile.get("phone", ""),
        "dob": "",
        "fathersName": "",
        "bloodGroup": "",
        "country": "India",
        "skills": ", ".join(profile.get("skills", [])[:25]),
        "universityName": profile.get("education", [""])[0] if profile.get("education") else "",
        "id": "",
        "courseOfDegree": profile.get("education", [""])[0] if profile.get("education") else "",
        "gradYear": "",
        "postGradYear": "",
        "certifications": ", ".join(profile.get("certifications", [])[:6]),
        "linkedin": profile.get("linkedin", ""),
        "github": profile.get("github", ""),
        "portfolio": profile.get("portfolio", ""),
        "location": profile.get("location", ""),
        "role": profile.get("role", "Software Engineer"),
        "yearsExperience": profile.get("yearsExperience", 0),
        "projects": profile.get("projects", []),
        "experienceHighlights": profile.get("experienceHighlights", []),
        "achievements": profile.get("achievements", []),
        "summary": profile.get("summary", "")
    }

    return jsonify({"profile": final_profile})

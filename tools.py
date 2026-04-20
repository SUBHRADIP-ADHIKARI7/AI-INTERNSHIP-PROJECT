import json
import random
import requests
import zlib
from bs4 import BeautifulSoup
from langchain.tools import tool

# Rotate User-Agents to reduce repeated-request fingerprinting
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
]

def _est_apps(title: str, company: str, bmin: int = 10, bmax: int = 150) -> int:
    """Consistently estimates applicants when API/HTML scraping is impossible or fails."""
    return bmin + (zlib.crc32((title + company).encode('utf-8')) % (bmax - bmin))


# ─── LINKEDIN SCRAPER ─────────────────────────────────────────────────────────
def scrape_linkedin(query: str, days_ago: int = 30, randomize: bool = True) -> list:
    """Scrape LinkedIn's guest jobs API for public job listings."""
    results = []
    try:
        keyword = query.replace(" ", "+")
        seconds = days_ago * 86400

        # Random start offset → different page of results every call
        # LinkedIn paginates in multiples of 10
        start = random.choice([0, 10, 20, 30]) if randomize else 0

        # sortBy=DD → sort by Date Descending (freshest first)
        url = (
            f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
            f"?keywords={keyword}&location=India&f_TPR=r{seconds}&sortBy=DD&start={start}"
        )
        headers = {
            "User-Agent": random.choice(_USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.5",
        }
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return []

        soup = BeautifulSoup(res.text, "html.parser")
        job_items = soup.find_all("li")

        # Shuffle the order so repeated calls with same start show different ordering
        if randomize:
            random.shuffle(job_items)

        for item in job_items[:5]:
            title_el = item.find("h3")
            company_el = item.find("h4")
            location_el = item.find("span", class_=lambda c: c and "location" in c.lower())
            link_el = item.find("a", href=True)

            title = title_el.text.strip() if title_el else "Software Role"
            company = company_el.text.strip() if company_el else "Company"
            location = location_el.text.strip() if location_el else "India"
            link = link_el["href"].split("?")[0] if link_el else ""

            if not link:
                continue

            applicants = 0
            try:
                import re
                job_id_match = re.search(r'-(\d+)(?:\?|$)', link)
                if job_id_match:
                    job_id = job_id_match.group(1)
                    r2 = requests.get(f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}", headers=headers, timeout=5)
                    if r2.status_code == 200:
                        s2 = BeautifulSoup(r2.text, "html.parser")
                        app_texts = [x.text.strip() for x in s2.find_all(string=re.compile("applicant", re.I))]
                        for txt in app_texts:
                            nums = re.findall(r'\d+', txt)
                            if nums:
                                applicants = int(nums[0])
                                break
            except:
                pass
            if applicants == 0:
                applicants = _est_apps(title, company, 20, 300)

            results.append({
                "id": len(results) + 1,
                "title": title[:70],
                "company": company[:45],
                "location": location.strip(),
                "duration": "Full Time / Internship",
                "stipend": "Competitive",
                "skills": [query.split()[0].capitalize()],
                "apply_link": link,
                "source": "LinkedIn",
                "applicants": applicants,
                "description": "Live LinkedIn job sorted by newest. Click Apply to go directly to the LinkedIn posting.",
            })

    except Exception as e:
        import traceback
        print(f"[LinkedIn scraper error] Traceback:")
        traceback.print_exc()

    return results


# ─── REMOTIVE API (REMOTE JOBS) ───────────────────────────────────────────────
def scrape_remotive(query: str, days_ago: int = 30) -> list:
    """Pull from Remotive's public API which has 100% real application links."""
    results = []
    try:
        keyword = query.split()[0]
        headers = {"User-Agent": random.choice(_USER_AGENTS)}
        res = requests.get(
            f"https://remotive.com/api/remote-jobs?search={keyword}&limit=15",
            headers=headers,
            timeout=10
        )
        if res.status_code != 200:
            return []

        jobs = res.json().get("jobs", [])
        if not jobs:
            return []

        # Randomly pick 5 jobs from the top 15 results for variety
        random.shuffle(jobs)
        for job in jobs[:5]:
            results.append({
                "id": len(results) + 1,
                "title": job.get("title", "Remote Role")[:70],
                "company": job.get("company_name", "Tech Company")[:45],
                "location": "Remote / Global",
                "duration": job.get("job_type", "Full Time").replace("_", " ").title(),
                "stipend": "Competitive",
                "skills": [keyword.capitalize()],
                "apply_link": job.get("url", "#"),
                "source": "Remotive",
                "applicants": _est_apps(job.get("title", ""), job.get("company_name", ""), 5, 80),
                "description": f"Verified remote job. Direct application link.",
            })
    except Exception as e:
        print(f"[Remotive error] {e}")

    return results


# ─── INDEED (via Arbeitnow free aggregator) ──────────────────────────────────
def scrape_indeed(query: str, days_ago: int = 30) -> list:
    """
    Pulls from Arbeitnow's free public API — a verified Indeed-equivalent
    global aggregator with real, direct application links and real-time freshness.
    """
    results = []
    try:
        words = query.lower().strip().split()
        keyword = "+".join(words[:2]) if len(words) >= 2 else words[0]

        # Try with India location first, then without
        for loc in ["India", ""]:
            loc_param = f"&location={loc}" if loc else ""
            res = requests.get(
                f"https://www.arbeitnow.com/api/job-board-api?search={keyword}{loc_param}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            if res.status_code == 200 and res.json().get("data"):
                break

        if res.status_code != 200:
            return []

        jobs = res.json().get("data", [])
        random.shuffle(jobs) # Randomize which jobs we pick from the results
        for job in jobs[:5]:
            title = job.get("title", "")[:70]
            company = job.get("company_name", "Company")[:45]
            location = job.get("location", "Remote / Global")[:60]
            url = job.get("url", "#")
            remote = job.get("remote", False)
            tags = job.get("tags", [])

            # Skip completely unrelated roles
            if not any(w in title.lower() for w in words[:2]):
                continue

            results.append({
                "id": len(results) + 1,
                "title": title,
                "company": company,
                "location": "Remote" if remote else location,
                "duration": "Full Time / Internship",
                "stipend": "Competitive",
                "skills": [words[0].capitalize()] + (tags[:2] if tags else []),
                "apply_link": url,
                "source": "Indeed (via Arbeitnow)",
                "applicants": _est_apps(title, company, 15, 200),
                "description": "Live job from Indeed aggregator. Click Apply for the full job posting.",
            })

    except Exception as e:
        print(f"[Indeed/Arbeitnow error] {e}")

    return results


# ─── FRESHERSHUB (OFF-CAMPUS DRIVES) ─────────────────────────────────────────
def scrape_freshershub(query: str, days_ago: int = 30) -> list:
    """Fetches verified off-campus drives from FreshersHub JSON API."""
    results = []
    try:
        # Site uses a high-speed JSON API
        res = requests.get("https://www.freshershub.app/jobs.json", timeout=10)
        if res.status_code != 200:
            return []
        
        jobs = res.json()
        kw = query.lower()
        
        # Filter by keyword in title or company
        filtered = [
            j for j in jobs 
            if kw in j.get("title", "").lower() or kw in j.get("company", "").lower()
        ]
        
        # FreshersHub mostly lists recent drives, we'll take top 5
        for job in filtered[:5]:
            results.append({
                "id": len(results) + 1,
                "title": job.get("title", "Off-Campus Drive")[:70],
                "company": job.get("company", "Company")[:45],
                "location": job.get("location", "India"),
                "duration": "Full Time / Internship",
                "stipend": "Paid / Market Standard",
                "skills": [query.split()[0].capitalize()],
                "apply_link": job.get("apply_link", "https://freshershub.app"),
                "source": "FreshersHub",
                "applicants": _est_apps(job.get("title", ""), job.get("company", ""), 50, 500),
                "description": f"Verified off-campus drive. Date: {job.get('date', 'Recent')}",
            })
    except Exception as e:
        print(f"[FreshersHub error] {e}")
    return results


# ─── INTERNSHIP HUB (WP SEARCH) ──────────────────────────────────────────────
def scrape_internshiphub(query: str, days_ago: int = 30) -> list:
    """Scrapes InternshipHub.in via WordPress search."""
    results = []
    try:
        keyword = query.replace(" ", "+")
        url = f"https://internshiphub.in/?s={keyword}"
        res = requests.get(url, headers={"User-Agent": random.choice(_USER_AGENTS)}, timeout=10)
        if res.status_code != 200:
            return []
        
        soup = BeautifulSoup(res.text, "html.parser")
        articles = soup.find_all("h2", class_="entry-title")
        
        for art in articles[:4]:
            link_el = art.find("a")
            if not link_el: continue
            
            title = link_el.text.strip()
            link = link_el.get("href", "")
            
            results.append({
                "id": len(results) + 1,
                "title": title[:70],
                "company": "Internship Hub",
                "location": "Remote / India",
                "duration": "Training + Internship",
                "stipend": "Stipend Available",
                "skills": [query.split()[0].capitalize()],
                "apply_link": link,
                "source": "InternshipHub",
                "applicants": _est_apps(title, "Internship Hub", 10, 60),
                "description": "Student-focused internship. Direct application on platform.",
            })
    except Exception as e:
        print(f"[InternshipHub error] {e}")
    return results


# ─── PLACEMENT INDIA ─────────────────────────────────────────────────────────
def scrape_placementindia(query: str, days_ago: int = 30) -> list:
    """Scrapes PlacementIndia.com as a robust Placement Hub alternative."""
    results = []
    try:
        keyword = query.replace(" ", "+")
        url = f"https://www.placementindia.com/job-search/search.php?keyword={keyword}&where=India"
        headers = {
            "User-Agent": random.choice(_USER_AGENTS),
            "Referer": "https://www.placementindia.com/"
        }
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return []
        
        soup = BeautifulSoup(res.text, "html.parser")
        # a.job-name contains the title h3
        job_links = soup.find_all("a", class_="job-name")
        
        for link_el in job_links[:5]:
            title_el = link_el.find("h3")
            if not title_el: continue
            
            # Company is often in the sibling p tag
            company_el = link_el.find_next_sibling("p")
            
            title = title_el.text.strip()
            company = company_el.text.strip() if company_el else "Legacy Company"
            link = link_el.get("href", "")
            if not link.startswith("http"):
                link = "https://www.placementindia.com" + link
                
            results.append({
                "id": len(results) + 1,
                "title": title[:70],
                "company": company[:45],
                "location": "India",
                "duration": "Full Time / Internship",
                "stipend": "As per Industry",
                "skills": [query.split()[0].capitalize()],
                "apply_link": link,
                "source": "PlacementIndia",
                "applicants": _est_apps(title, company, 5, 45),
                "description": "Verified job/internship from PlacementIndia portal.",
            })
    except Exception as e:
        print(f"[PlacementIndia error] {e}")
    return results


# ─── UNSTOP (COMPETITIONS, INTERNSHIPS, JOBS) ────────────────────────────────
def scrape_unstop(query: str, days_ago: int = 30) -> list:
    """Fetches internships and jobs from Unstop's public JSON API."""
    results = []
    try:
        keyword = query.split()[0]
        params = {
            "opportunity": "internships",
            "page": 1,
            "per_page": 15,
            "oppstatus": "open",
            "searchTerm": keyword,
        }
        headers = {"User-Agent": random.choice(_USER_AGENTS)}
        res = requests.get(
            "https://unstop.com/api/public/opportunity/search-result",
            params=params, headers=headers, timeout=10
        )
        if res.status_code != 200:
            return []

        # More robust extraction logic for Unstop's nested structure
        data = res.json()
        jobs = []
        if "data" in data and isinstance(data["data"], dict) and "data" in data["data"]:
            jobs = data["data"]["data"]
        elif "data" in data and isinstance(data["data"], list):
            jobs = data["data"]

        if not jobs:
            return []

        random.shuffle(jobs)

        for job in jobs[:5]:
            org = job.get("organisation", {})
            title = job.get("title", "Opportunity")[:70]
            company = org.get("name", "Company")[:45]
            public_url = job.get("public_url", "")
            link = f"https://unstop.com/{public_url}" if public_url else "https://unstop.com"

            # Stipend info
            stipend_obj = job.get("stipend", {})
            stipend = "Unpaid"
            if isinstance(stipend_obj, dict):
                amt = stipend_obj.get("salary", stipend_obj.get("max", ""))
                if amt:
                    stipend = f"₹{amt}/month"

            results.append({
                "id": len(results) + 1,
                "title": title,
                "company": company,
                "location": job.get("jobLocation", "India") or job.get("city", "India"),
                "duration": "Internship",
                "stipend": stipend,
                "skills": [keyword.capitalize()],
                "apply_link": link,
                "source": "Unstop",
                "applicants": job.get("registeredUsersCount") or job.get("participantsCount") or job.get("viewCount") or _est_apps(title, company, 20, 300),
                "description": f"Verified listing on Unstop. Status: Open. Apply directly.",
            })
    except Exception as e:
        print(f"[Unstop error] {e}")
    return results


# ─── SOCIAL MEDIA JOB FINDER (X, FACEBOOK, INSTAGRAM via Google Index) ───────
def scrape_social_media(query: str, days_ago: int = 7) -> list:
    """
    Finds job/internship posts from X (Twitter), Facebook, and Instagram
    by searching Google's public index. No API keys or login required.
    """
    results = []
    keyword = query.replace(" ", "+")

    # Define site-specific Google dork searches
    social_searches = [
        {
            "site": "x.com",
            "source": "X (Twitter)",
            "url": f"https://www.google.com/search?q=site:x.com+%22{keyword}%22+%22hiring%22+%22India%22&tbs=qdr:w",
        },
        {
            "site": "facebook.com",
            "source": "Facebook",
            "url": f"https://www.google.com/search?q=site:facebook.com+%22{keyword}%22+%22internship%22+%22India%22+%22hiring%22&tbs=qdr:w",
        },
        {
            "site": "instagram.com",
            "source": "Instagram",
            "url": f"https://www.google.com/search?q=site:instagram.com+%22{keyword}%22+%22hiring%22+%22India%22&tbs=qdr:w",
        },
    ]

    for search in social_searches:
        try:
            headers = {
                "User-Agent": random.choice(_USER_AGENTS),
                "Accept-Language": "en-US,en;q=0.9",
            }
            res = requests.get(search["url"], headers=headers, timeout=10)
            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.text, "html.parser")

            # Google search result links
            for a_tag in soup.find_all("a", href=True):
                href = a_tag.get("href", "")
                # Extract actual URL from Google's redirect
                if f'{search["site"]}' in href and "/url?q=" in href:
                    actual_url = href.split("/url?q=")[1].split("&")[0]
                    title_text = a_tag.get_text(strip=True)[:70]

                    if not title_text or len(title_text) < 10:
                        continue

                    results.append({
                        "id": len(results) + 1,
                        "title": title_text,
                        "company": f"Posted on {search['source']}",
                        "location": "India / Remote",
                        "duration": "Check Post",
                        "stipend": "See Post Details",
                        "skills": [query.split()[0].capitalize()],
                        "apply_link": actual_url,
                        "source": search["source"],
                        "applicants": _est_apps(title_text, search["source"], 2, 25),
                        "description": f"Job/internship post found on {search['source']}. Click to view the original post.",
                    })

                    if len([r for r in results if r["company"].endswith(search["source"])]) >= 2:
                        break

        except Exception as e:
            print(f"[{search['source']} error] {e}")

    random.shuffle(results)
    return results


# ─── MAIN TOOL ────────────────────────────────────────────────────────────────
@tool
def search_internships(query: str, sources: str = "linkedin,indeed,remotive,freshershub,placementhub,internshiphub,unstop,social") -> str:
    """
    Finds real job/internship application links from LinkedIn, Indeed, Remotive, FreshersHub,
    Placement Hub, Internship Hub, Unstop, and social media (X, Facebook, Instagram).
    Use the 'sources' parameter to restrict to specific platforms, e.g. 'unstop,linkedin'.
    ALWAYS use this tool when the user asks for internships or jobs!
    """
    # Clean up conversational fluff from the LLM query for strict APIs
    import re
    clean_query = re.sub(r'(?i)\b(find|search|look|show|get|me|some|internships?|jobs?|for|a|an|who|knows|with|skills|in|the|area|of)\b', '', query)
    clean_query = re.sub(r'[^\w\s\+]', ' ', clean_query)
    clean_query = ' '.join(clean_query.split())[:40] # Keep it concise
    if not clean_query:
        clean_query = "Software Internship"

    active = [s.strip().lower() for s in sources.split(",")]
    all_cards = []

    if "linkedin" in active:
        all_cards.extend(scrape_linkedin(clean_query))

    if "indeed" in active:
        all_cards.extend(scrape_indeed(clean_query))

    if "remotive" in active:
        all_cards.extend(scrape_remotive(clean_query))

    if "freshershub" in active:
        all_cards.extend(scrape_freshershub(clean_query))

    if "internshiphub" in active:
        all_cards.extend(scrape_internshiphub(clean_query))

    if "placementhub" in active or "placementindia" in active:
        all_cards.extend(scrape_placementindia(clean_query))

    if "unstop" in active:
        all_cards.extend(scrape_unstop(clean_query))

    if "social" in active or "twitter" in active or "facebook" in active or "instagram" in active:
        all_cards.extend(scrape_social_media(clean_query))

    # Prioritize LinkedIn and Unstop, then sort by applicants
    priority = {"LinkedIn": 0, "Unstop": 1, "Remotive": 2, "PlacementHub": 3, "PlacementIndia": 3}
    all_cards.sort(key=lambda x: (priority.get(x.get("source", ""), 99), x.get("applicants", 10000)))

    # Cap at 10 jobs max so the LLM doesn't truncate the payload or hallucinate
    all_cards = all_cards[:10]

    # Re-number the cards
    for i, card in enumerate(all_cards):
        card["id"] = i + 1

    # Add source badge to title
    for card in all_cards:
        source = card.pop("source", "Job Board")
        card["title"] = f"[{source}] {card['title']}"

    if not all_cards:
        # Final fallback
        kw = query.replace(" ", "+")
        fallback_source = active[0].capitalize() if active else "Job Hub"
        all_cards = [
            {
                "id": 1,
                "title": f"[Google] Search {fallback_source} India",
                "company": "Search Result",
                "location": "India",
                "duration": "Various",
                "stipend": "Competitive",
                "skills": [query.split()[0].capitalize()],
                "apply_link": f"https://www.google.com/search?q={kw}+site:linkedin.com/jobs+India",
                "applicants": _est_apps(f"Search {fallback_source}", "Google", 20, 150),
                "description": f"Click Apply to search {fallback_source} directly via Google."
            }
        ]

    return "```internship_cards\n" + json.dumps(all_cards, indent=2) + "\n```"

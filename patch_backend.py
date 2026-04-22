"""Fix: add fallback data when scraping returns no results."""
import ast, sys

lines = open('app.py', encoding='utf-8').read().splitlines(keepends=True)
print(f"File has {len(lines)} lines")

# Find the "skillGap" line in the return jsonify block
ret_idx = None
for i, line in enumerate(lines):
    if '"skillGap": gap_skills,' in line or '"skillGap":gap_skills,' in line:
        ret_idx = i
        break

if ret_idx is None:
    print("ERROR: skillGap return line not found")
    # Show the return section
    for i, l in enumerate(lines[655:680], 656):
        print(f"  {i}: {l.rstrip()[:90]}")
    sys.exit(1)

print(f"Found skillGap return at line {ret_idx+1}")

# Find the "for job in top_jobs:" line (Step 8 loop)
step8_end = None
for i in range(ret_idx, max(ret_idx-10, 0), -1):
    if "job['insight']" in lines[i] or "job[\"insight\"]" in lines[i]:
        step8_end = i
        break

if step8_end is None:
    print("ERROR: step8 end not found"); sys.exit(1)

# Insert fallback code right after step8_end loop (before return jsonify)
# Find the blank line before "return jsonify"
insert_at = step8_end + 1
while insert_at < ret_idx and lines[insert_at].strip() == '':
    insert_at += 1
# insert_at is now the line of "return jsonify"
print(f"Inserting fallback code at line {insert_at+1} (before return jsonify)")

FALLBACK_CODE = """\
        # Fallback: static market trends when scraping returns nothing
        if not market_trends:
            market_trends = [
                {'role': 'Software Engineer', 'count': 45, 'percent': 35},
                {'role': 'Data Science',      'count': 26, 'percent': 20},
                {'role': 'Machine Learning',  'count': 20, 'percent': 16},
                {'role': 'Web Development',   'count': 18, 'percent': 14},
                {'role': 'Mobile Dev',        'count': 10, 'percent': 8},
                {'role': 'DevOps / Cloud',    'count': 9,  'percent': 7},
            ]

        # Fallback: role-based gap skills when scraping returns nothing
        if not gap_skills:
            ROLE_GAP_MAP = {
                'Software Engineer':   ['Docker','Kubernetes','AWS','System Design','GraphQL','CI/CD'],
                'Frontend Developer':  ['TypeScript','Next.js','GraphQL','Jest','Docker','AWS'],
                'Data Scientist':      ['PyTorch','TensorFlow','Spark','Airflow','MLOps','Docker'],
                'Machine Learning':    ['PyTorch','MLOps','Docker','Kubernetes','Spark','GCP'],
                'Full Stack Developer':['Docker','AWS','GraphQL','Redis','Kubernetes','CI/CD'],
                'Mobile Developer':    ['Kotlin','Swift','React Native','Firebase','CI/CD','AWS'],
                'DevOps Engineer':     ['Terraform','Ansible','GCP','Azure','Prometheus','Grafana'],
            }
            fallback_role = inferred_role if inferred_role in ROLE_GAP_MAP else 'Software Engineer'
            for sk in ROLE_GAP_MAP.get(fallback_role, []):
                if sk.lower() not in user_skills_lower:
                    gap_skills.append({'skill': sk, 'demand': 3, 'priority': 'high'})
            # Also add medium-priority generic skills
            generic = ['Docker','AWS','Git','Agile','REST API','PostgreSQL','Redis','CI/CD']
            for sk in generic:
                if sk.lower() not in user_skills_lower and not any(g['skill']==sk for g in gap_skills):
                    gap_skills.append({'skill': sk, 'demand': 2, 'priority': 'medium'})
                if len(gap_skills) >= 10:
                    break

"""

result_lines = lines[:insert_at] + [FALLBACK_CODE] + lines[insert_at:]
open('app.py', 'w', encoding='utf-8', newline='\r\n').writelines(result_lines)
print(f"Written. New file has {len(result_lines)} lines")

# Syntax check
try:
    ast.parse(open('app.py', encoding='utf-8').read())
    print("Syntax OK - SUCCESS!")
except SyntaxError as e:
    print(f"SYNTAX ERROR at line {e.lineno}: {e.msg}")
    all_lines = open('app.py', encoding='utf-8').read().splitlines()
    for i, l in enumerate(all_lines[max(0,e.lineno-4):e.lineno+3], max(1,e.lineno-3)):
        print(f"  {i}: {l[:90]}")

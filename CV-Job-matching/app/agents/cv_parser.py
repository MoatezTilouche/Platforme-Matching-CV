# app/agents/cv_parser.py
import ollama

def parse_cv(cv_text: str):
    prompt = f"""Extract structured info from CV. Return JSON only.

Fields:
- skills: list of technical skills
- years_experience: number
- job_titles: list of recent titles
- domains: list of industries/domains
- seniority: junior|mid|senior|lead

CV:
{cv_text[:2000]}
"""

    res = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={
            "num_predict": 300,
            "temperature": 0.1
        }
    )

    return res["message"]["content"]

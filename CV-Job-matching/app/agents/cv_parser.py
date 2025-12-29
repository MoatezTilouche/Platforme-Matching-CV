# app/agents/cv_parser.py
from ollama import AsyncClient

async def parse_cv(cv_text: str):
    prompt = f"""Extract from CV as JSON:

{{
  "skills": ["skill1", "skill2"],
  "years_experience": 5,
  "job_titles": ["title1"],
  "domains": ["domain1"],
  "seniority": "mid"
}}

CV:
{cv_text[:1500]}
"""

    client = AsyncClient()
    res = await client.chat(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={
            "num_predict": 200,
            "temperature": 0.1
        }
    )

    return res["message"]["content"]

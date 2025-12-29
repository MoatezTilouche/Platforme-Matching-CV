# app/agents/jd_parser.py
from ollama import AsyncClient


async def parse_jd(jd_text: str):
    prompt = f"""Extract requirements as JSON:

{{
  "required_skills": [],
  "nice_to_have": [],
  "min_years_experience": 3,
  "domain": [],
  "seniority": "mid"
}}

JD:
{jd_text[:1500]}
"""

    client = AsyncClient()
    res = await client.chat(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={
            "num_predict": 150,
            "temperature": 0.1
        }
    )

    return res["message"]["content"]

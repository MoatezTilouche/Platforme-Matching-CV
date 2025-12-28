# app/agents/jd_parser.py
import ollama


def parse_jd(jd_text: str):
    prompt = f"""Extract job requirements. Return JSON only.

Fields:
- required_skills: list
- nice_to_have: list
- min_years_experience: number
- domain: list
- seniority: junior|mid|senior|lead

JD:
{jd_text[:2000]}
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

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from typing import List

# Load Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load assessment dataset
df = pd.read_csv("SHLTask1.csv")

# Request body
class QueryInput(BaseModel):
    query: str

# FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "SHL Assessment API is running."}

@app.post("/recommend")
def recommend_assessments(data: QueryInput):
    model = genai.GenerativeModel("gemini-1.5-pro-001")

    prompt = f"""
You are an assistant helping HRs select SHL assessments.

Here is a job description:
\"\"\"{data.query}\"\"\"

Suggest 1–10 relevant SHL assessments based on this job description.

Use the following list of assessments:
{df.to_string(index=False)}

Return only a valid JSON list in this format:
[
  {{
    "url": "https://...",
    "adaptive_support": "Yes" or "No",
    "description": "Text",
    "duration": 30,
    "remote_support": "Yes" or "No",
    "test_type": ["Type1", "Type2"]
  }},
  ...
]
Only return the list — no extra text or explanation.
    """

    try:
        response = model.generate_content(prompt)
        assessments = json.loads(response.text)

        formatted = []
        for item in assessments:
            formatted.append({
                "url": item.get("url", ""),
                "adaptive_support": item.get("adaptive_support", "No"),
                "description": item.get("description", ""),
                "duration": int(item.get("duration", 0)),
                "remote_support": item.get("remote_support", "No"),
                "test_type": item.get("test_type", []) or []
            })
            if len(formatted) == 10:
                break

        return {"recommended_assessments": formatted}

    except Exception as e:
        return {
            "recommended_assessments": [],
            "error": str(e)
        }

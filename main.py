from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import pandas as pd
import os
import json
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

df = pd.read_csv("SHLTask1.csv")

app = FastAPI()

class QueryInput(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "API is running!", "usage": "POST to /recommend"}

@app.post("/recommend")
def recommend_assessments(input: QueryInput):
    prompt = f"""
You’re an expert assistant helping recruiters choose SHL assessments for this:
\"\"\"{input.query}\"\"\"

Pick up to 10 from:
{df.to_string(index=False)}

Return only a JSON list like this:
[
  {{
    "url": "https://...",
    "adaptive_support": "Yes",
    "description": "Test description",
    "duration": 45,
    "remote_support": "Yes",
    "test_type": ["Technical", "Behavioral"]
  }}
]
"""

    model = genai.GenerativeModel("gemini-1.5-pro-001")
    response = model.generate_content(prompt)

    try:
        result = json.loads(response.text)
        return {"recommended_assessments": result}
    except json.JSONDecodeError:
        return {"error": "Could not parse response", "raw": response.text}

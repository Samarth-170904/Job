from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

genai.configure(api_key=api_key)

# Load dataset
df = pd.read_csv("SHLTask1.csv")

# Define expected input model
class QueryInput(BaseModel):
    query: str

# Define output format
class AssessmentOutput(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: list[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: list[AssessmentOutput]

# FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "SHL Assessment API is running."}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(data: QueryInput):
    model = genai.GenerativeModel("gemini-1.5-pro-001")

    prompt = f"""
You’re an expert assistant helping recruiters choose SHL assessments.

Given this job description:
\"\"\"{data.query}\"\"\"

Return up to 10 relevant assessments from the list:
{df.to_string(index=False)}

Format the response **exactly** like this:
[
  {{
    "url": "https://...",
    "adaptive_support": "Yes",
    "description": "Text here",
    "duration": 45,
    "remote_support": "Yes",
    "test_type": ["Category1", "Category2"]
  }},
  ...
]
Only return the JSON list.
"""

    try:
        response = model.generate_content(prompt)
        parsed = json.loads(response.text)

        # Validate and limit results to max 10
        validated = []
        for item in parsed:
            validated.append({
                "url": item.get("url", ""),
                "adaptive_support": item.get("adaptive_support", "No"),
                "description": item.get("description", ""),
                "duration": int(item.get("duration", 0)),
                "remote_support": item.get("remote_support", "No"),
                "test_type": item.get("test_type", []),
            })
            if len(validated) == 10:
                break

        return {"recommended_assessments": validated}

    except json.JSONDecodeError:
        return {
            "recommended_assessments": [],
            "error": "Failed to parse Gemini response",
            "raw_response": response.text
        }

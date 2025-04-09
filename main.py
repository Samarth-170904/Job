from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# Load environment variable from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Validate API key
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please add it to the .env or Render environment.")

genai.configure(api_key=api_key)

# Load SHL Assessments Data
df = pd.read_csv("SHLTask1.csv")

# Define Request Schema
class QueryInput(BaseModel):
    query: str

# Create FastAPI instance
app = FastAPI()

@app.get("/")
def root():
    return {"message": "SHL Assessment Recommendation API is running."}

@app.post("/recommend")
def recommend_assessments(data: QueryInput):
    model = genai.GenerativeModel("gemini-1.5-pro-001")

    prompt = f"""
You’re an expert assistant helping recruiters and HR professionals choose the best SHL assessments for hiring.

Here’s a job description or requirement:
\"\"\"{data.query}\"\"\"

Based on this, suggest up to 10 relevant assessments from the list below.

Available assessments:
{df.to_string(index=False)}

For each recommendation, return a JSON object with:
- "Name" (name of the assessment),
- "URL" (link to the assessment),
- "Duration" (in minutes),
- "Type" (e.g. Cognitive, Behavioral, Technical),
- "Remote" (Yes or No),
- "Adaptive" (Yes or No)

Please return only the JSON list like this:
[
  {{
    "Name": "Assessment Name",
    "URL": "https://...",
    "Duration": 45,
    "Type": "Cognitive",
    "Remote": "Yes",
    "Adaptive": "Yes"
  }},
  ...
]
"""

    response = model.generate_content(prompt)

    try:
        results = json.loads(response.text)
        return {"results": results}
    except json.JSONDecodeError:
        return {
            "error": "Failed to parse Gemini model response",
            "raw_response": response.text
        }

import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from agents.parser import ResumeData  
from agents.pdf_loader import extract_text_from_pdf  
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Missing GOOGLE_API_KEY in environment variables")


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=google_api_key,
    temperature=0.1
)
structured_model = model.with_structured_output(ResumeData)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  

@app.post("/parse-resume/")
async def parse_resume(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        
        pdf_text = extract_text_from_pdf(file.file)

        
        prompt = f"""
        Extract structured resume data from the following text.

        Return the data as JSON in this format:
        - personal_info: includes name, email, phone, location
        - work_experience: list of companies, titles, durations, responsibilities
        - education: list of degrees, institutions, graduation years
        - skills: list of skills
        - tech_stack: list of technologies
        - achievements: list of achievements
        - certifications: list of certifications

        Resume Text:
        {pdf_text}
        """

        result = structured_model.invoke(prompt)
        return JSONResponse(content=result.dict())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def getRequest():
    return {"message": "Welcome to the Resume Parser API!"}

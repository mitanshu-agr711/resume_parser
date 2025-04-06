from pydantic import BaseModel, Field
from typing import List

class PersonalInfo(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address")
    phone: str = Field(description="Phone number")
    location: str = Field(description="Address or city")

class WorkExperience(BaseModel):
    company: str
    title: str
    duration: str
    responsibilities: List[str]

class Education(BaseModel):
    degree: str
    institution: str
    graduation_year: str

class ResumeData(BaseModel):
    personal_info: PersonalInfo
    work_experience: List[WorkExperience]
    education: List[Education]
    skills: List[str]
    tech_stack: List[str]
    achievements: List[str]
    certifications: List[str]

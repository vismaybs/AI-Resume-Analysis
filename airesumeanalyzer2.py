from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
from pathlib import Path
import uuid

from src.parser import ResumeParser
from src.analyzer import ResumeAnalyzer

app = FastAPI(title="AI Resume Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = ResumeParser()
analyzer = ResumeAnalyzer()

class AnalysisResponse(BaseModel):
    resume_id: str
    overall_score: float
    ats_score: float
    skill_match_percentage: float
    matched_skills: List[str]
    missing_skills: List[str]
    feedback: List[str]
    formatting_analysis: dict
    extracted_skills: List[str]
    experience_years: float

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_resume(
    file: UploadFile = File(...),
    job_requirements: Optional[str] = None
):
    """Analyze a resume file (PDF or DOCX)"""
    
    # Validate file type
    allowed_extensions = ['.pdf', '.docx']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type. Use {allowed_extensions}")
    
    # Save temporary file
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    temp_path = temp_dir / f"{uuid.uuid4()}{file_extension}"
    
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse resume
        parsed_resume = parser.parse_resume(str(temp_path))
        
        # Parse job requirements if provided
        job_skills = []
        if job_requirements:
            job_skills = parser.extract_skills(job_requirements)
        
        # Analyze
        analysis = analyzer.full_analysis(parsed_resume, job_skills)
        
        return AnalysisResponse(
            resume_id=str(uuid.uuid4()),
            overall_score=analysis['overall_score'],
            ats_score=analysis['ats_score'],
            skill_match_percentage=analysis['skill_match'].get('match_percentage', 0),
            matched_skills=analysis['skill_match'].get('matched_skills', []),
            missing_skills=analysis['skill_match'].get('missing_skills', []),
            feedback=analysis['feedback'],
            formatting_analysis=analysis['formatting'],
            extracted_skills=parsed_resume['skills'],
            experience_years=analysis['experience_years']
        )
    
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "AI Resume Analyzer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
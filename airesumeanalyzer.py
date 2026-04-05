from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime

class ResumeAnalyzer:
    """Professional resume analyzer with scoring and recommendations"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.industry_keywords = self._load_industry_keywords()
    
    def _load_industry_keywords(self) -> Dict:
        """Load industry-specific keywords"""
        return {
            'software_engineering': [
                'agile', 'scrum', 'git', 'ci/cd', 'microservices', 'api',
                'cloud', 'devops', 'testing', 'debugging', 'optimization'
            ],
            'data_science': [
                'statistics', 'analytics', 'visualization', 'prediction',
                'modeling', 'hypothesis', 'a/b testing', 'data mining'
            ],
            'project_management': [
                'leadership', 'stakeholder', 'budget', 'timeline', 'risk',
                'resource', 'milestone', 'deliverable', 'scheduling'
            ]
        }
    
    def calculate_ats_score(self, resume_text: str) -> float:
        """Calculate ATS compatibility score"""
        score = 0
        max_score = 100
        
        # Check for standard sections
        required_sections = ['experience', 'education', 'skills', 'contact']
        section_patterns = {
            'experience': r'(work|professional|employment|experience)',
            'education': r'(education|academic|university|college)',
            'skills': r'(skills|technical|competencies)',
            'contact': r'(email|phone|address|contact)'
        }
        
        for section, pattern in section_patterns.items():
            if re.search(pattern, resume_text, re.IGNORECASE):
                score += 15
        
        # Check for proper formatting (no tables, columns, graphics)
        if not re.search(r'\|', resume_text):  # No table structures
            score += 10
        
        # Check for standard fonts and clean text
        if len(resume_text.split()) > 300:  # Sufficient content
            score += 10
        
        # Check for action verbs
        action_verbs = ['managed', 'developed', 'created', 'led', 'implemented', 
                       'designed', 'built', 'achieved', 'improved', 'increased']
        verbs_found = sum(1 for verb in action_verbs if verb in resume_text.lower())
        score += min(verbs_found, 10)
        
        # Check for quantifiable achievements
        quantifiers = re.findall(r'\d+%|\d+\s*(?:dollars|USD|years|months)', resume_text)
        score += min(len(quantifiers) * 2, 20)
        
        return min(score, max_score)
    
    def calculate_skill_match(self, resume_skills: List[str], job_requirements: List[str]) -> Dict:
        """Calculate skill match percentage"""
        if not job_requirements:
            return {'match_percentage': 0, 'matched_skills': [], 'missing_skills': []}
        
        resume_skills_lower = [s.lower() for s in resume_skills]
        job_skills_lower = [s.lower() for s in job_requirements]
        
        matched = [skill for skill in job_requirements 
                  if skill.lower() in resume_skills_lower]
        missing = [skill for skill in job_requirements 
                  if skill.lower() not in resume_skills_lower]
        
        match_percentage = (len(matched) / len(job_requirements)) * 100
        
        return {
            'match_percentage': round(match_percentage, 2),
            'matched_skills': matched,
            'missing_skills': missing
        }
    
    def analyze_formatting(self, resume_text: str) -> Dict:
        """Analyze resume formatting and structure"""
        lines = resume_text.split('\n')
        
        # Check bullet points usage
        bullet_lines = [line for line in lines if line.strip().startswith(('•', '-', '*'))]
        
        # Check paragraph length
        avg_line_length = np.mean([len(line) for line in lines if line.strip()])
        
        # Check section headers
        headers = re.findall(r'^[A-Z][A-Z\s]+$', resume_text, re.MULTILINE)
        
        return {
            'bullet_points_count': len(bullet_lines),
            'average_line_length': round(avg_line_length, 2),
            'section_headers_count': len(headers),
            'has_contact_section': bool(re.search(r'contact', resume_text, re.IGNORECASE)),
            'has_summary': bool(re.search(r'(summary|profile|about)', resume_text, re.IGNORECASE))
        }
    
    def generate_feedback(self, analysis: Dict) -> List[str]:
        """Generate actionable feedback for improvement"""
        feedback = []
        
        # ATS feedback
        ats_score = analysis.get('ats_score', 0)
        if ats_score < 60:
            feedback.append("⚠️ Low ATS score. Add standard sections (Experience, Education, Skills)")
        elif ats_score < 80:
            feedback.append("✓ Good ATS compatibility. Consider adding more quantifiable achievements")
        
        # Skill feedback
        skill_match = analysis.get('skill_match', {}).get('match_percentage', 0)
        if skill_match < 50:
            feedback.append("⚠️ Low skill match. Add missing key skills from job description")
        
        # Formatting feedback
        formatting = analysis.get('formatting', {})
        if formatting.get('bullet_points_count', 0) < 5:
            feedback.append("💡 Use more bullet points to highlight achievements")
        
        if not formatting.get('has_summary', False):
            feedback.append("💡 Add a professional summary at the top")
        
        # Experience feedback
        experience_years = analysis.get('experience_years', 0)
        if experience_years == 0:
            feedback.append("💡 Highlight your work experience with dates")
        elif experience_years < 2:
            feedback.append("💡 Emphasize internships and projects if you're early-career")
        
        # Add positive feedback
        if skill_match > 80:
            feedback.append("🎉 Excellent skill match! Your resume is well-aligned")
        
        if ats_score > 85:
            feedback.append("🏆 Great ATS optimization!")
        
        return feedback
    
    def full_analysis(self, parsed_resume: Dict, job_requirements: List[str] = None) -> Dict:
        """Perform complete resume analysis"""
        resume_text = parsed_resume['raw_text']
        
        # Calculate various scores
        ats_score = self.calculate_ats_score(resume_text)
        formatting = self.analyze_formatting(resume_text)
        
        skill_match = {}
        if job_requirements:
            skill_match = self.calculate_skill_match(
                parsed_resume['skills'], 
                job_requirements
            )
        
        # Calculate overall score
        overall_score = (
            ats_score * 0.4 +
            skill_match.get('match_percentage', 0) * 0.4 +
            (formatting['bullet_points_count'] * 2) * 0.2
        )
        
        analysis = {
            'ats_score': ats_score,
            'skill_match': skill_match,
            'formatting': formatting,
            'overall_score': round(overall_score, 2),
            'experience_years': parsed_resume['experience']['estimated_experience'],
            'skills_found': parsed_resume['skills'],
            'word_count': parsed_resume['word_count']
        }
        
        analysis['feedback'] = self.generate_feedback(analysis)
        
        return analysis
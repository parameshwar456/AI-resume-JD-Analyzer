import PyPDF2
import docx
import re
from typing import Dict, List
import spacy

class ResumeParser:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Skills database
        self.tech_skills = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust',
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'fastapi',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd',
            'machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch',
            'html', 'css', 'sass', 'tailwind', 'bootstrap', 'rest api', 'graphql', 'microservices'
        ]
    
    def parse_resume(self, file_path: str) -> Dict:
        """Parse resume and extract information"""
        
        # Extract text
        if file_path.endswith('.pdf'):
            text = self._extract_pdf(file_path)
        elif file_path.endswith('.docx'):
            text = self._extract_docx(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Parse data
        return {
            'raw_text': text,
            'contact': self._extract_contact(text),
            'skills': self._extract_skills(text),
            'experience': self._extract_experience(text),
            'education': self._extract_education(text),
            'total_years': self._calculate_experience(text),
            'word_count': len(text.split())
        }
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    
    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = docx.Document(file_path)
        return '\n'.join([p.text for p in doc.paragraphs])
    
    def _extract_contact(self, text: str) -> Dict:
        """Extract contact information"""
        # Email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        # Phone
        phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
        phones = re.findall(phone_pattern, text)
        
        # LinkedIn
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text, re.IGNORECASE)
        
        # GitHub
        github_pattern = r'github\.com/[\w-]+'
        github = re.findall(github_pattern, text, re.IGNORECASE)
        
        return {
            'email': emails[0] if emails else None,
            'phone': phones[0] if phones else None,
            'linkedin': linkedin[0] if linkedin else None,
            'github': github[0] if github else None
        }
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical skills"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.tech_skills:
            if skill in text_lower:
                # Capitalize properly
                found_skills.append(skill.title())
        
        return list(set(found_skills))
    
    def _extract_experience(self, text: str) -> List[str]:
        """Extract work experience"""
        # Look for experience section
        exp_pattern = r'(?i)(experience|work history|employment)(.*?)(?=education|skills|projects|$)'
        match = re.search(exp_pattern, text, re.DOTALL)
        
        if match:
            exp_text = match.group(2)
            # Split by job entries
            jobs = re.split(r'\n(?=[A-Z])', exp_text)
            return [job.strip() for job in jobs if len(job.strip()) > 50]
        
        return []
    
    def _extract_education(self, text: str) -> List[str]:
        """Extract education"""
        edu_pattern = r'(?i)(education|academic)(.*?)(?=experience|skills|projects|certifications|$)'
        match = re.search(edu_pattern, text, re.DOTALL)
        
        if match:
            edu_text = match.group(2)
            degrees = re.findall(r'(bachelor|master|phd|b\.tech|m\.tech|b\.sc|m\.sc|mba|bba).*', edu_text, re.IGNORECASE)
            return degrees
        
        return []
    
    def _calculate_experience(self, text: str) -> int:
        """Calculate total years of experience"""
        # Find year ranges
        year_pattern = r'(\d{4})\s*[-–]\s*(\d{4}|present|current)'
        matches = re.findall(year_pattern, text.lower())
        
        total_years = 0
        current_year = 2024
        
        for match in matches:
            start = int(match[0])
            end = current_year if match[1] in ['present', 'current'] else int(match[1])
            total_years += max(0, end - start)
        
        return total_years
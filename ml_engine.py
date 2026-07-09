import numpy as np
import re
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class MLEngine:
    def __init__(self):
        # Load Sentence Transformer (BERT-based)
        print("Loading AI models...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ BERT model loaded")
        
        # TF-IDF
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Skills database
        self.all_skills = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'react', 'angular',
            'vue', 'node.js', 'django', 'flask', 'spring', 'sql', 'mysql', 'postgresql',
            'mongodb', 'aws', 'azure', 'docker', 'kubernetes', 'git', 'machine learning',
            'deep learning', 'nlp', 'tensorflow', 'pytorch', 'rest api', 'graphql'
        ]
    
    def analyze(self, resume_data: Dict, job_description: str, job_title: str) -> Dict:
        """Perform comprehensive AI analysis"""
        
        print("Starting analysis...")
        
        results = {
            'ats_score': self._calculate_ats_score(resume_data, job_description),
            'match_score': self._calculate_match_score(resume_data, job_description),
            'semantic_score': self._calculate_semantic_score(resume_data, job_description),
            'keyword_match': self._analyze_keywords(resume_data, job_description),
            'matched_keywords': self._get_matched_keywords(resume_data, job_description),
            'missing_keywords': self._get_missing_keywords(resume_data, job_description),
            'skill_gap': self._analyze_skill_gap(resume_data, job_description),
            'experience': self._analyze_experience(resume_data, job_description),
            'education': self._analyze_education(resume_data),
            'suggestions': self._generate_suggestions(resume_data, job_description),
            'ai_summary': self._generate_summary(resume_data),
            'salary': self._predict_salary(resume_data, job_title),
            'interview_questions': self._generate_interview_questions(resume_data)
        }
        
        print("Analysis complete!")
        return results
    
    def _calculate_ats_score(self, resume_data: Dict, job_description: str) -> float:
        """Calculate ATS score using ML"""
        score = 100.0
        
        # Check contact info
        if not resume_data['contact']['email']:
            score -= 15
        if not resume_data['contact']['phone']:
            score -= 10
        
        # Check sections
        if not resume_data['skills']:
            score -= 20
        if not resume_data['experience']:
            score -= 20
        if not resume_data['education']:
            score -= 15
        
        # Keyword matching
        jd_lower = job_description.lower()
        resume_text = resume_data['raw_text'].lower()
        
        jd_words = set(re.findall(r'\b\w+\b', jd_lower))
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        jd_keywords = jd_words - stop_words
        
        resume_words = set(re.findall(r'\b\w+\b', resume_text))
        matched = len(resume_words.intersection(jd_keywords))
        
        keyword_score = (matched / len(jd_keywords) * 20) if jd_keywords else 0
        score = score - 20 + keyword_score
        
        return max(0, min(100, score))
    
    def _calculate_match_score(self, resume_data: Dict, job_description: str) -> float:
        """Calculate overall match using TF-IDF"""
        try:
            resume_text = resume_data['raw_text']
            
            # Fit TF-IDF
            tfidf_matrix = self.tfidf.fit_transform([resume_text, job_description])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            return float(similarity[0][0] * 100)
        except:
            return 0.0
    
    def _calculate_semantic_score(self, resume_data: Dict, job_description: str) -> float:
        """Calculate semantic similarity using BERT"""
        resume_text = str(resume_data.get('raw_text') or '').strip()
        job_text = str(job_description or '').strip()

        if not resume_text or not job_text:
            return 0.0

        try:
            # Generate embeddings
            resume_embedding = self.sentence_model.encode(resume_text, convert_to_tensor=False)
            jd_embedding = self.sentence_model.encode(job_text, convert_to_tensor=False)

            # Calculate cosine similarity
            similarity = np.dot(resume_embedding, jd_embedding) / (
                np.linalg.norm(resume_embedding) * np.linalg.norm(jd_embedding)
            )

            return float(similarity * 100)
        except Exception:
            return 0.0
    
    def _analyze_keywords(self, resume_data: Dict, job_description: str) -> Dict:
        """Analyze keyword matching"""
        resume_skills = set([s.lower() for s in resume_data['skills']])
        jd_skills = self._extract_skills_from_jd(job_description)
        
        matched = resume_skills.intersection(jd_skills)
        missing = jd_skills - resume_skills
        
        return {
            'matched': len(matched),
            'missing': len(missing),
            'total': len(jd_skills),
            'percentage': round((len(matched) / len(jd_skills) * 100), 1) if jd_skills else 0
        }
    
    def _get_matched_keywords(self, resume_data: Dict, job_description: str) -> List[str]:
        """Get list of matched keywords"""
        resume_skills = set([s.lower() for s in resume_data['skills']])
        jd_skills = self._extract_skills_from_jd(job_description)
        matched = resume_skills.intersection(jd_skills)
        return sorted(list(matched))
    
    def _get_missing_keywords(self, resume_data: Dict, job_description: str) -> List[str]:
        """Get list of missing keywords"""
        resume_skills = set([s.lower() for s in resume_data['skills']])
        jd_skills = self._extract_skills_from_jd(job_description)
        missing = jd_skills - resume_skills
        return sorted(list(missing))
    
    def _extract_skills_from_jd(self, job_description: str) -> set:
        """Extract skills from job description"""
        jd_lower = job_description.lower()
        found = set()
        
        for skill in self.all_skills:
            if skill in jd_lower:
                found.add(skill)
        
        return found
    
    def _analyze_skill_gap(self, resume_data: Dict, job_description: str) -> Dict:
        """Analyze skill gaps with priority"""
        resume_skills = set([s.lower() for s in resume_data['skills']])
        jd_skills = self._extract_skills_from_jd(job_description)
        missing = jd_skills - resume_skills
        
        jd_lower = job_description.lower()
        
        critical = []
        important = []
        nice_to_have = []
        
        for skill in missing:
            count = jd_lower.count(skill)
            if count >= 3:
                critical.append(skill)
            elif count == 2:
                important.append(skill)
            else:
                nice_to_have.append(skill)
        
        return {
            'critical': critical,
            'important': important,
            'nice_to_have': nice_to_have
        }
    
    def _analyze_experience(self, resume_data: Dict, job_description: str) -> Dict:
        """Analyze experience fit"""
        total_years = resume_data['total_years']
        
        # Extract required experience from JD
        required_pattern = r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience'
        matches = re.findall(required_pattern, job_description.lower())
        required_years = int(matches[0]) if matches else 0
        
        return {
            'total_years': total_years,
            'required_years': required_years,
            'meets_requirement': total_years >= required_years
        }
    
    def _analyze_education(self, resume_data: Dict) -> Dict:
        """Analyze education"""
        education = resume_data['education']
        
        level = "None"
        if education:
            edu_text = ' '.join(education).lower()
            if 'phd' in edu_text or 'doctorate' in edu_text:
                level = "Doctorate"
            elif 'master' in edu_text or 'm.tech' in edu_text:
                level = "Master's"
            elif 'bachelor' in edu_text or 'b.tech' in edu_text:
                level = "Bachelor's"
        
        return {
            'level': level,
            'count': len(education),
            'score': 100 if education else 50
        }
    
    def _generate_suggestions(self, resume_data: Dict, job_description: str) -> List[Dict]:
        """Generate AI-powered suggestions"""
        suggestions = []
        
        # Contact
        if not resume_data['contact']['email']:
            suggestions.append({
                'category': 'Contact',
                'priority': 'Critical',
                'suggestion': 'Add professional email address'
            })
        
        if not resume_data['contact']['linkedin']:
            suggestions.append({
                'category': 'Contact',
                'priority': 'High',
                'suggestion': 'Add LinkedIn profile URL'
            })
        
        # Skills
        skill_gap = self._analyze_skill_gap(resume_data, job_description)
        if skill_gap['critical']:
            suggestions.append({
                'category': 'Skills',
                'priority': 'Critical',
                'suggestion': f"Add critical skills: {', '.join(skill_gap['critical'][:3])}"
            })
        
        # Experience
        exp = self._analyze_experience(resume_data, job_description)
        if not exp['meets_requirement'] and exp['required_years'] > 0:
            suggestions.append({
                'category': 'Experience',
                'priority': 'High',
                'suggestion': f"Job requires {exp['required_years']} years, you have {exp['total_years']}"
            })
        
        # Education
        if not resume_data['education']:
            suggestions.append({
                'category': 'Education',
                'priority': 'Medium',
                'suggestion': 'Add education details'
            })
        
        # Word count
        if resume_data['word_count'] < 300:
            suggestions.append({
                'category': 'Content',
                'priority': 'High',
                'suggestion': 'Resume is too short. Add more details.'
            })
        
        return suggestions
    
    def _generate_summary(self, resume_data: Dict) -> str:
        """Generate AI summary"""
        skills_count = len(resume_data['skills'])
        years = resume_data['total_years']
        
        summary = f"Candidate with {years} years of experience"
        
        if skills_count > 0:
            summary += f" and {skills_count} technical skills"
        
        if resume_data['education']:
            summary += f", holding {len(resume_data['education'])} degree(s)"
        
        summary += "."
        
        return summary
    
    def _predict_salary(self, resume_data: Dict, job_title: str) -> Dict:
        """Predict salary using ML model"""
        years = resume_data['total_years']
        skills = len(resume_data['skills'])
        has_degree = len(resume_data['education']) > 0
        
        # Salary prediction logic (simplified ML model)
        if years < 2:
            base = (50000, 75000)
            level = "Entry Level"
        elif years < 5:
            base = (75000, 110000)
            level = "Mid Level"
        elif years < 10:
            base = (110000, 160000)
            level = "Senior Level"
        else:
            base = (150000, 220000)
            level = "Lead Level"
        
        # Modifiers
        modifier = 1.0
        if has_degree:
            modifier += 0.1
        if skills >= 15:
            modifier += 0.1
        if skills >= 10:
            modifier += 0.05
        
        min_sal = int(base[0] * modifier)
        max_sal = int(base[1] * modifier)
        avg_sal = (min_sal + max_sal) // 2
        
        # Confidence based on data completeness
        confidence = 50
        if resume_data['contact']['email']:
            confidence += 10
        if resume_data['experience']:
            confidence += 15
        if resume_data['education']:
            confidence += 15
        if skills >= 10:
            confidence += 10
        
        return {
            'min': min_sal,
            'avg': avg_sal,
            'max': max_sal,
            'level': level,
            'confidence': min(100, confidence)
        }
    
    def _generate_interview_questions(self, resume_data: Dict) -> Dict:
        """Generate interview questions"""
        
        hr_questions = [
            "Tell me about yourself and your background.",
            "Why do you want to work for our company?",
            "What are your greatest strengths?",
            "What are your weaknesses?",
            "Where do you see yourself in 5 years?",
            "Why should we hire you?",
            "Describe a challenging situation you faced.",
            "How do you handle stress and pressure?"
        ]
        
        technical_questions = [
            "Explain your experience with the technologies on your resume.",
            "Describe your most complex project.",
            "How do you approach problem-solving?",
            "What is your development workflow?",
            "How do you ensure code quality?",
            "Explain a technical challenge you overcame.",
            "How do you stay updated with new technologies?",
            "Describe your experience with version control."
        ]
        
        resume_based = [
            f"Can you elaborate on your {resume_data['total_years']} years of experience?",
            "Tell me about your most recent role.",
            "What was your biggest achievement?",
            "Why are you looking for a new opportunity?"
        ]
        
        if resume_data['skills']:
            resume_based.append(f"How would you rate your proficiency in {resume_data['skills'][0]}?")
        
        return {
            'hr': hr_questions,
            'technical': technical_questions,
            'resume_based': resume_based
        }
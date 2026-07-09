# AI Resume-JD Analyzer (Streamlit + ML Version)

A comprehensive AI-powered resume analyzer built entirely with **Python**, **Streamlit**, and **Advanced Machine Learning models**.

## 🎯 Features

- **Resume Parsing**: Extract structured data from PDF/DOCX using spaCy NLP
- **ATS Score**: Calculate Applicant Tracking System compatibility (0-100)
- **Semantic Matching**: BERT-based resume-JD similarity using Sentence Transformers
- **TF-IDF Analysis**: Traditional text similarity scoring
- **Skill Gap Analysis**: Identify missing skills with priority classification
- **Grammar Check**: LanguageTool integration for error detection
- **Readability Analysis**: Flesch Reading Ease, Gunning Fog, SMOG Index
- **Salary Prediction**: ML-based salary estimation
- **Interview Questions**: AI-generated HR, behavioral, and technical questions
- **Visual Analytics**: Interactive Plotly charts and dashboards

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **NLP**: spaCy (en_core_web_lg)
- **Semantic Similarity**: Sentence Transformers (all-mpnet-base-v2)
- **ML**: Scikit-learn, XGBoost
- **Grammar**: LanguageTool
- **Readability**: Textstat
- **Visualization**: Plotly
- **Document Parsing**: PyPDF2, python-docx

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/ai-resume-analyzer-ml.git
cd ai-resume-analyzer-ml


# Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
```

2. Create a virtual environment:

**Windows**
```bash
python -m venv venv
```

3. Activate the virtual environment:

**Command Prompt**
```bash
venv\Scripts\activate
```

**PowerShell**
```powershell
.\venv\Scripts\Activate.ps1
```

4. Install the required packages:

```bash
pip install -r requirements.txt
```

5. Run the project:

```bash
python app.py
```
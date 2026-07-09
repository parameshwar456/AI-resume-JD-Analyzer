import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
import os

from resume_parser import ResumeParser
from ml_engine import MLEngine

# Page config
st.set_page_config(
    page_title="AI Resume-JD Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .score-excellent {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .score-good {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    .score-average {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .score-poor {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.3rem;
    }
    
    .badge-success { background: #10b981; color: white; }
    .badge-danger { background: #ef4444; color: white; }
    .badge-warning { background: #f59e0b; color: white; }
    .badge-info { background: #3b82f6; color: white; }
    
    h1, h2, h3 {
        color: #1f2937;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f4f6;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'parser' not in st.session_state:
    st.session_state.parser = ResumeParser()
if 'ml_engine' not in st.session_state:
    st.session_state.ml_engine = MLEngine()

# Sidebar
with st.sidebar:
    st.markdown("# 📄 AI Resume Analyzer")
    st.markdown("---")
    
    st.markdown("### 🤖 AI Features")
    st.markdown("""
    - ✅ NLP Resume Parsing
    - ✅ BERT Semantic Matching
    - ✅ ML Salary Prediction
    - ✅ Skill Gap Analysis
    - ✅ ATS Score (0-100)
    - ✅ Interview Questions
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Models Used")
    st.markdown("""
    - **spaCy**: NLP parsing
    - **BERT**: Semantic similarity
    - **XGBoost**: Salary prediction
    - **TF-IDF**: Keyword matching
    """)
    
    st.markdown("---")
    st.info("💡 Upload your resume and paste a job description to get started!")

# Main app
st.title("🎯 AI-Powered Resume-JD Analyzer")
st.markdown("### Optimize your resume with Machine Learning and Natural Language Processing")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results", "📈 Detailed Insights"])

# TAB 1: Upload & Analyze
with tab1:
    st.markdown("## Step 1: Upload Your Resume")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF or DOCX file",
            type=['pdf', 'docx'],
            help="Upload your resume in PDF or DOCX format (Max 10MB)"
        )
        
        if uploaded_file:
            # Save file temporarily
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Parse resume
            with st.spinner("🔍 Parsing your resume with AI..."):
                try:
                    resume_data = st.session_state.parser.parse_resume(file_path)
                    st.session_state.resume_data = resume_data
                    
                    # Delete temp file
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    st.success("✅ Resume parsed successfully!")
                    
                    # Show preview
                    with st.expander("👁️ View Parsed Data", expanded=True):
                        col_a, col_b, col_c, col_d = st.columns(4)
                        
                        with col_a:
                            st.metric("📧 Email", "Found" if resume_data['contact']['email'] else "Not Found")
                        with col_b:
                            st.metric("🛠️ Skills", len(resume_data['skills']))
                        with col_c:
                            st.metric("💼 Experience", f"{resume_data['total_years']} years")
                        with col_d:
                            st.metric("📝 Words", resume_data['word_count'])
                        
                        st.markdown("#### Contact Information")
                        st.json(resume_data['contact'])
                        
                        st.markdown("#### Skills Found")
                        if resume_data['skills']:
                            skills_html = " ".join([f"<span class='badge badge-info'>{skill}</span>" for skill in resume_data['skills'][:15]])
                            st.markdown(skills_html, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"❌ Error parsing resume: {str(e)}")
    
    with col2:
        st.markdown("### 📋 Tips")
        st.info("""
        **For best results:**
        - Use clear section headers
        - List skills clearly
        - Include contact info
        - Avoid images/graphics
        - Keep it 1-2 pages
        """)
    
    st.markdown("---")
    st.markdown("## Step 2: Paste Job Description")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        job_title = st.text_input(
            "🎯 Job Title",
            placeholder="e.g., Senior Software Engineer"
        )
        
        job_description = st.text_area(
            "📝 Job Description",
            height=250,
            placeholder="Paste the complete job description here including requirements, responsibilities, and qualifications..."
        )
    
    with col2:
        if job_description:
            word_count = len(job_description.split())
            st.metric("JD Words", word_count)
            
            if word_count < 50:
                st.warning("⚠️ JD seems short. Add more details for better analysis.")
            else:
                st.success("✅ Good JD length")
    
    st.markdown("---")
    
    # Analyze button
    if st.button("🚀 Analyze with AI", disabled=not (uploaded_file and job_description)):
        if not st.session_state.resume_data:
            st.error("❌ Please upload a resume first!")
        elif not job_description:
            st.error("❌ Please paste a job description!")
        else:
            # Progress bar
            progress_bar = st.progress(0)
            status = st.empty()
            
            try:
                # Step 1
                status.text("⚙️ Initializing ML models...")
                progress_bar.progress(20)
                
                # Step 2
                status.text("🔍 Calculating ATS score...")
                progress_bar.progress(40)
                
                # Step 3
                status.text("🤖 Running BERT semantic analysis...")
                progress_bar.progress(60)
                
                # Step 4
                status.text("📊 Analyzing skill gaps...")
                progress_bar.progress(80)
                
                # Perform analysis
                results = st.session_state.ml_engine.analyze(
                    st.session_state.resume_data,
                    job_description,
                    job_title
                )
                
                st.session_state.analysis_results = results
                
                # Complete
                status.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                st.success("🎉 Analysis completed successfully!")
                st.balloons()
                
                # Auto switch to results tab
                st.info("👉 Click on the **Results** tab to view your analysis!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)

# TAB 2: Results
with tab2:
    if st.session_state.analysis_results is None:
        st.warning("⚠️ No analysis available. Please upload a resume and analyze it first.")
    else:
        results = st.session_state.analysis_results
        
        st.markdown("## 📊 Analysis Results")
        
        # Main Scores
        st.markdown("### 🎯 Overall Scores")
        col1, col2, col3, col4 = st.columns(4)
        
        def get_score_class(score):
            if score >= 80: return 'score-excellent'
            elif score >= 60: return 'score-good'
            elif score >= 40: return 'score-average'
            else: return 'score-poor'
        
        def get_score_label(score):
            if score >= 80: return 'Excellent ⭐'
            elif score >= 60: return 'Good ✓'
            elif score >= 40: return 'Average ⚠'
            else: return 'Poor ✗'
        
        with col1:
            ats = results['ats_score']
            st.markdown(f"""
            <div class='metric-card {get_score_class(ats)}'>
                <h2 style='margin:0;'>{ats:.1f}%</h2>
                <p style='margin:0; font-size:0.9rem;'>ATS Score</p>
                <p style='margin:0; font-size:0.8rem; opacity:0.9;'>{get_score_label(ats)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            match = results['match_score']
            st.markdown(f"""
            <div class='metric-card {get_score_class(match)}'>
                <h2 style='margin:0;'>{match:.1f}%</h2>
                <p style='margin:0; font-size:0.9rem;'>Match Score</p>
                <p style='margin:0; font-size:0.8rem; opacity:0.9;'>{get_score_label(match)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            semantic = results['semantic_score']
            st.markdown(f"""
            <div class='metric-card {get_score_class(semantic)}'>
                <h2 style='margin:0;'>{semantic:.1f}%</h2>
                <p style='margin:0; font-size:0.9rem;'>BERT Score</p>
                <p style='margin:0; font-size:0.8rem; opacity:0.9;'>AI Match</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            keyword = results['keyword_match']['percentage']
            st.markdown(f"""
            <div class='metric-card {get_score_class(keyword)}'>
                <h2 style='margin:0;'>{keyword:.1f}%</h2>
                <p style='margin:0; font-size:0.9rem;'>Keywords</p>
                <p style='margin:0; font-size:0.8rem; opacity:0.9;'>{results['keyword_match']['matched']}/{results['keyword_match']['total']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        st.markdown("### 📈 Visual Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Donut chart
            fig = go.Figure(data=[go.Pie(
                labels=['ATS', 'Match', 'BERT', 'Keywords'],
                values=[ats, match, semantic, keyword],
                hole=0.4,
                marker_colors=['#667eea', '#10b981', '#f59e0b', '#8b5cf6']
            )])
            fig.update_layout(
                title="Score Distribution",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart for keyword analysis
            fig = go.Figure(data=[
                go.Bar(name='Matched', x=['Keywords'], y=[results['keyword_match']['matched']], marker_color='#10b981'),
                go.Bar(name='Missing', x=['Keywords'], y=[results['keyword_match']['missing']], marker_color='#ef4444')
            ])
            fig.update_layout(
                title="Keyword Analysis",
                barmode='group',
                height=400,
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Keyword Details
        st.markdown("### 🔍 Keyword Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Matched Keywords")
            if results['matched_keywords']:
                keywords_html = " ".join([f"<span class='badge badge-success'>{kw}</span>" for kw in results['matched_keywords']])
                st.markdown(keywords_html, unsafe_allow_html=True)
            else:
                st.warning("No matched keywords")
        
        with col2:
            st.markdown("#### ❌ Missing Keywords")
            if results['missing_keywords']:
                keywords_html = " ".join([f"<span class='badge badge-danger'>{kw}</span>" for kw in results['missing_keywords'][:20]])
                st.markdown(keywords_html, unsafe_allow_html=True)
                if len(results['missing_keywords']) > 20:
                    st.info(f"... and {len(results['missing_keywords']) - 20} more")
            else:
                st.success("No missing keywords!")
        
        st.markdown("---")
        
        # Skill Gap
        st.markdown("### 🎯 Skill Gap Analysis")
        
        gap = results['skill_gap']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔴 Critical Skills")
            if gap['critical']:
                for skill in gap['critical'][:5]:
                    st.markdown(f"- {skill}")
            else:
                st.success("None")
        
        with col2:
            st.markdown("#### 🟡 Important Skills")
            if gap['important']:
                for skill in gap['important'][:5]:
                    st.markdown(f"- {skill}")
            else:
                st.success("None")
        
        with col3:
            st.markdown("#### 🟢 Nice to Have")
            if gap['nice_to_have']:
                for skill in gap['nice_to_have'][:5]:
                    st.markdown(f"- {skill}")
            else:
                st.success("None")
        
        st.markdown("---")
        
        # Suggestions
        st.markdown("### 💡 AI Suggestions")
        
        for suggestion in results['suggestions']:
            priority_color = {
                'Critical': 'danger',
                'High': 'warning',
                'Medium': 'info',
                'Low': 'success'
            }
            
            st.markdown(f"""
            <div style='background:#f9fafb; padding:1rem; border-radius:10px; margin:0.5rem 0; border-left:4px solid {
                '#ef4444' if suggestion['priority'] == 'Critical' else
                '#f59e0b' if suggestion['priority'] == 'High' else
                '#3b82f6' if suggestion['priority'] == 'Medium' else '#10b981'
            }'>
                <strong><span class='badge badge-{priority_color[suggestion["priority"]]}'>{suggestion['priority']}</span> {suggestion['category']}</strong>
                <p style='margin:0.5rem 0 0 0;'>{suggestion['suggestion']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Salary Prediction
        st.markdown("### 💰 ML Salary Prediction")
        
        salary = results['salary']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card' style='background:linear-gradient(135deg, #10b981 0%, #059669 100%); color:white;'>
                <h3 style='margin:0;'>${salary['min']:,}</h3>
                <p style='margin:0;'>Minimum</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white;'>
                <h3 style='margin:0;'>${salary['avg']:,}</h3>
                <p style='margin:0;'>Average</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card' style='background:linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color:white;'>
                <h3 style='margin:0;'>${salary['max']:,}</h3>
                <p style='margin:0;'>Maximum</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.info(f"**Level:** {salary['level']} | **Confidence:** {salary['confidence']:.0f}%")

# TAB 3: Detailed Insights
with tab3:
    if st.session_state.analysis_results is None:
        st.warning("⚠️ No analysis available.")
    else:
        results = st.session_state.analysis_results
        
        st.markdown("## 📈 Detailed Analysis Report")
        
        # Interview Questions
        st.markdown("### 🎤 Interview Preparation")
        
        q_tab1, q_tab2, q_tab3 = st.tabs(["HR Questions", "Technical Questions", "Resume-Based"])
        
        with q_tab1:
            st.markdown("#### 👔 HR Interview Questions")
            for i, q in enumerate(results['interview_questions']['hr'], 1):
                st.markdown(f"{i}. {q}")
        
        with q_tab2:
            st.markdown("#### 💻 Technical Questions")
            for i, q in enumerate(results['interview_questions']['technical'], 1):
                st.markdown(f"{i}. {q}")
        
        with q_tab3:
            st.markdown("#### 📄 Resume-Based Questions")
            for i, q in enumerate(results['interview_questions']['resume_based'], 1):
                st.markdown(f"{i}. {q}")
        
        st.markdown("---")
        
        # Experience Analysis
        st.markdown("### 📅 Experience Analysis")
        
        exp = results['experience']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Your Experience", f"{exp['total_years']} years")
        with col2:
            st.metric("Required", f"{exp['required_years']} years")
        with col3:
            st.metric("Status", "✅ Meets" if exp['meets_requirement'] else "❌ Below")
        
        # Progress bar
        if exp['required_years'] > 0:
            percentage = min(100, (exp['total_years'] / exp['required_years']) * 100)
            st.progress(percentage / 100)
            st.caption(f"Experience coverage: {percentage:.0f}%")
        
        st.markdown("---")
        
        # Education Analysis
        st.markdown("### 🎓 Education Analysis")
        
        edu = results['education']
        
        st.metric("Education Score", f"{edu['score']:.0f}%")
        st.write(f"**Level:** {edu['level']}")
        st.write(f"**Degrees:** {edu['count']}")
        
        st.markdown("---")
        
        # AI Summary
        st.markdown("### 🤖 AI-Generated Summary")
        st.info(results['ai_summary'])
        
        st.markdown("---")
        
        # Download Report
        st.markdown("### 📥 Download Report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'scores': {
                'ats_score': results['ats_score'],
                'match_score': results['match_score'],
                'semantic_score': results['semantic_score']
            },
            'keyword_analysis': results['keyword_match'],
            'skill_gap': results['skill_gap'],
            'salary_prediction': results['salary'],
            'suggestions': results['suggestions']
        }
        
        json_report = json.dumps(report, indent=2)
        
        st.download_button(
            label="📥 Download JSON Report",
            data=json_report,
            file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#6b7280; padding:2rem 0;'>
    <p>Built with ❤️ using Streamlit, spaCy, BERT, and XGBoost</p>
    <p style='font-size:0.85rem;'>AI Resume-JD Analyzer v1.0 | © 2024</p>
</div>
""", unsafe_allow_html=True)
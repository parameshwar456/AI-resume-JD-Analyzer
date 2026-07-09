import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / "trained_models"
DATA_DIR = BASE_DIR / "data"
RESUMES_DIR = DATA_DIR / "resumes"
JD_DIR = DATA_DIR / "job_descriptions"

# Create directories
for directory in [MODELS_DIR, DATA_DIR, RESUMES_DIR, JD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ML Model Configurations
SPACY_MODEL = "en_core_web_lg"
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
BERT_MODEL = "bert-base-uncased"

# Skill Database - Comprehensive Tech Skills
TECH_SKILLS_DB = {
    "programming_languages": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Ruby", "PHP",
        "Go", "Rust", "Swift", "Kotlin", "Scala", "R", "MATLAB", "Perl", "Dart",
        "Elixir", "Haskell", "Lua", "Shell", "Bash", "PowerShell"
    ],
    "web_frontend": [
        "React", "Angular", "Vue.js", "Svelte", "Next.js", "Nuxt.js", "Gatsby",
        "HTML5", "CSS3", "Sass", "SCSS", "Less", "Tailwind CSS", "Bootstrap",
        "Material-UI", "Ant Design", "Chakra UI", "Styled Components", "jQuery",
        "Ember.js", "Backbone.js", "Alpine.js", "Webpack", "Vite", "Parcel"
    ],
    "web_backend": [
        "Node.js", "Express.js", "Django", "Flask", "FastAPI", "Spring Boot",
        "ASP.NET", "Ruby on Rails", "Laravel", "Symfony", "CodeIgniter", "NestJS",
        "Koa", "Hapi", "Fastify", "Phoenix", "Gin", "Echo", "Fiber"
    ],
    "mobile": [
        "React Native", "Flutter", "Swift", "Kotlin", "Xamarin", "Ionic",
        "Cordova", "SwiftUI", "Jetpack Compose", "Android Studio", "Xcode"
    ],
    "databases": [
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra", "DynamoDB",
        "Oracle", "SQL Server", "SQLite", "MariaDB", "CouchDB", "Neo4j",
        "InfluxDB", "Firebase", "Supabase", "Elasticsearch", "Memcached",
        "RocksDB", "ScyllaDB", "ArangoDB"
    ],
    "cloud_platforms": [
        "AWS", "Azure", "Google Cloud", "GCP", "Heroku", "DigitalOcean",
        "IBM Cloud", "Oracle Cloud", "Alibaba Cloud", "Vercel", "Netlify",
        "CloudFlare", "Linode", "Vultr"
    ],
    "devops": [
        "Docker", "Kubernetes", "Jenkins", "GitLab CI", "GitHub Actions",
        "CircleCI", "Travis CI", "Terraform", "Ansible", "Chef", "Puppet",
        "ArgoCD", "Spinnaker", "Bamboo", "TeamCity", "Drone", "Vagrant",
        "Packer", "Helm", "Istio", "Prometheus", "Grafana", "ELK Stack",
        "Datadog", "New Relic", "Nagios", "Consul", "Vault"
    ],
    "ai_ml": [
        "Machine Learning", "Deep Learning", "Natural Language Processing",
        "Computer Vision", "TensorFlow", "PyTorch", "Keras", "Scikit-learn",
        "OpenCV", "YOLO", "Pandas", "NumPy", "SciPy", "Matplotlib", "Seaborn",
        "Hugging Face", "spaCy", "NLTK", "Gensim", "LangChain", "LlamaIndex",
        "MLflow", "Weights & Biases", "Kubeflow", "Airflow", "Ray", "Dask",
        "JAX", "MXNet", "Caffe", "Detectron2", "Stable Diffusion", "GPT",
        "BERT", "Transformers", "XGBoost", "LightGBM", "CatBoost"
    ],
    "data_engineering": [
        "Apache Spark", "Hadoop", "Kafka", "Airflow", "Databricks", "Snowflake",
        "Redshift", "BigQuery", "Flink", "Storm", "Presto", "Hive", "Pig",
        "Sqoop", "Flume", "NiFi", "dbt", "Fivetran", "Stitch", "Talend"
    ],
    "version_control": [
        "Git", "GitHub", "GitLab", "Bitbucket", "SVN", "Mercurial", "Perforce"
    ],
    "testing": [
        "Jest", "Mocha", "Chai", "Jasmine", "Pytest", "JUnit", "TestNG",
        "Selenium", "Cypress", "Playwright", "Puppeteer", "Robot Framework",
        "Cucumber", "Postman", "SoapUI", "JMeter", "Locust", "K6"
    ],
    "methodologies": [
        "Agile", "Scrum", "Kanban", "DevOps", "CI/CD", "TDD", "BDD",
        "Microservices", "REST API", "GraphQL", "gRPC", "WebSockets",
        "Serverless", "Event-Driven", "Domain-Driven Design", "SOLID",
        "Design Patterns", "Clean Architecture"
    ]
}

# Salary Data by Role and Experience (USD)
SALARY_DATA = {
    "entry_level": {  # 0-2 years
        "software_engineer": (55000, 85000),
        "data_scientist": (65000, 95000),
        "ml_engineer": (70000, 100000),
        "frontend_developer": (50000, 75000),
        "backend_developer": (55000, 80000),
        "fullstack_developer": (55000, 85000),
        "mobile_developer": (55000, 80000),
        "devops_engineer": (60000, 90000),
        "cloud_engineer": (60000, 90000),
        "data_engineer": (60000, 90000),
        "default": (50000, 75000)
    },
    "mid_level": {  # 2-5 years
        "software_engineer": (85000, 130000),
        "data_scientist": (95000, 150000),
        "ml_engineer": (100000, 160000),
        "frontend_developer": (75000, 110000),
        "backend_developer": (80000, 120000),
        "fullstack_developer": (85000, 130000),
        "mobile_developer": (80000, 120000),
        "devops_engineer": (90000, 135000),
        "cloud_engineer": (90000, 135000),
        "data_engineer": (90000, 135000),
        "default": (75000, 110000)
    },
    "senior_level": {  # 5-10 years
        "software_engineer": (130000, 200000),
        "data_scientist": (150000, 220000),
        "ml_engineer": (160000, 240000),
        "frontend_developer": (110000, 160000),
        "backend_developer": (120000, 180000),
        "fullstack_developer": (130000, 200000),
        "mobile_developer": (120000, 180000),
        "devops_engineer": (135000, 195000),
        "cloud_engineer": (135000, 195000),
        "data_engineer": (135000, 195000),
        "default": (110000, 170000)
    },
    "lead_level": {  # 10+ years
        "software_engineer": (170000, 280000),
        "data_scientist": (190000, 320000),
        "ml_engineer": (200000, 350000),
        "frontend_developer": (140000, 220000),
        "backend_developer": (160000, 250000),
        "fullstack_developer": (170000, 280000),
        "mobile_developer": (160000, 250000),
        "devops_engineer": (165000, 260000),
        "cloud_engineer": (165000, 260000),
        "data_engineer": (165000, 260000),
        "default": (150000, 240000)
    }
}

# Interview Questions Database
INTERVIEW_QUESTIONS = {
    "hr": [
        "Tell me about yourself and your professional background.",
        "Why do you want to work for our company?",
        "What are your greatest strengths and weaknesses?",
        "Where do you see yourself in 5 years?",
        "Why should we hire you over other candidates?",
        "Describe a challenging situation you faced and how you handled it.",
        "How do you handle stress and pressure in the workplace?",
        "What is your expected salary range?",
        "What motivates you in your career?",
        "Do you have any questions for us?"
    ],
    "behavioral": [
        "Describe a time when you had to work with a difficult team member.",
        "Tell me about a time when you failed and what you learned.",
        "Give an example of a situation where you had to meet a tight deadline.",
        "Tell me about a time when you showed leadership skills.",
        "Describe a situation where you had to learn something new quickly.",
        "How do you prioritize tasks when working on multiple projects?",
        "Tell me about a time you received constructive criticism.",
        "Describe a situation where you went above and beyond.",
        "How do you handle conflicts in the workplace?",
        "Tell me about your biggest professional achievement."
    ],
    "technical": {
        "python": [
            "Explain the difference between list and tuple in Python.",
            "What are decorators and how do you use them?",
            "Explain the Global Interpreter Lock (GIL) in Python.",
            "What is the difference between deep copy and shallow copy?",
            "Explain list comprehension and generator expressions.",
            "What are Python's magic methods or dunder methods?",
            "Explain multithreading vs multiprocessing in Python.",
            "What is a context manager and how do you create one?",
            "Explain the difference between staticmethod and classmethod.",
            "How does garbage collection work in Python?"
        ],
        "javascript": [
            "Explain closures in JavaScript with an example.",
            "What is the difference between let, const, and var?",
            "Explain event delegation and event bubbling.",
            "What are Promises and how do they work?",
            "Explain async/await and how it differs from Promises.",
            "What is the event loop in JavaScript?",
            "Explain the concept of hoisting.",
            "What is the difference between == and ===?",
            "Explain prototypal inheritance in JavaScript.",
            "What are arrow functions and how do they differ from regular functions?"
        ],
        "react": [
            "What are React hooks and why were they introduced?",
            "Explain the Virtual DOM and how React uses it.",
            "What is the difference between state and props?",
            "Explain the useEffect hook and its use cases.",
            "What are higher-order components (HOCs)?",
            "Explain React context API and when to use it.",
            "What is the difference between controlled and uncontrolled components?",
            "Explain React reconciliation process.",
            "What are custom hooks and how do you create them?",
            "Explain React's component lifecycle methods."
        ],
        "machine_learning": [
            "Explain the difference between supervised and unsupervised learning.",
            "What is overfitting and how do you prevent it?",
            "Explain bias-variance tradeoff.",
            "What is the difference between L1 and L2 regularization?",
            "Explain the working of Random Forest algorithm.",
            "What is gradient descent and its variants?",
            "Explain precision, recall, and F1-score.",
            "What is the curse of dimensionality?",
            "Explain the difference between bagging and boosting.",
            "What is cross-validation and why is it important?"
        ],
        "databases": [
            "Explain the difference between SQL and NoSQL databases.",
            "What is database normalization and its types?",
            "Explain ACID properties in databases.",
            "What are indexes and how do they improve query performance?",
            "Explain different types of JOIN operations.",
            "What is the difference between DELETE and TRUNCATE?",
            "Explain database transactions and isolation levels.",
            "What are triggers and stored procedures?",
            "Explain sharding and replication in databases.",
            "What is the CAP theorem?"
        ],
        "algorithms": [
            "Explain time and space complexity with Big O notation.",
            "What is the difference between Stack and Queue?",
            "Explain binary search algorithm and its complexity.",
            "What are different sorting algorithms and their complexities?",
            "Explain dynamic programming with an example.",
            "What is the difference between BFS and DFS?",
            "Explain hash tables and collision resolution techniques.",
            "What are greedy algorithms? Give examples.",
            "Explain the concept of recursion and memoization.",
            "What is a trie data structure and its applications?"
        ],
        "system_design": [
            "How would you design a URL shortener like bit.ly?",
            "Design a rate limiter for an API.",
            "How would you design Instagram or Twitter?",
            "Explain how you would design a distributed cache.",
            "Design a real-time chat application.",
            "How would you design a ride-sharing service like Uber?",
            "Explain load balancing strategies.",
            "How would you design a recommendation system?",
            "Design a video streaming service like Netflix.",
            "How would you handle millions of concurrent users?"
        ]
    }
}

# ATS Scoring Weights
ATS_WEIGHTS = {
    "keywords": 0.30,
    "formatting": 0.20,
    "sections": 0.20,
    "experience": 0.15,
    "education": 0.10,
    "length": 0.05
}

# Streamlit Page Config
PAGE_CONFIG = {
    "page_title": "AI Resume-JD Analyzer",
    "page_icon": "📄",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Color Scheme
COLORS = {
    "primary": "#2563eb",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#3b82f6",
    "purple": "#8b5cf6"
}
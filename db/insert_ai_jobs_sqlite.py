import sqlite3

# Representative job postings data (from PerplexityAI, early 2025)
jobs_data = [
    {
        "job_id": 1,
        "title": "AI Engineer",
        "company": "Major Tech Firm (Google/Microsoft)",
        "location": "London/Remote",
        "salary": "£70,000 - £120,000+",
        "description": "Design, develop AI solutions; work with ML models, neural networks; contribute to AI product roadmap",
        "ai_impact": "High demand role; core to AI innovation",
        "date_posted": "Feb 2025",
        "source": "[1][4][5]"
    },
    {
        "job_id": 2,
        "title": "Machine Learning Specialist",
        "company": "Fintech Startup",
        "location": "London",
        "salary": "£65,000 - £100,000",
        "description": "Develop ML algorithms for financial data; collaborate with data scientists; improve model accuracy",
        "ai_impact": "AI-driven automation focus",
        "date_posted": "Early 2025",
        "source": "[1][5]"
    },
    {
        "job_id": 3,
        "title": "Data Scientist",
        "company": "Cloud Services Company",
        "location": "Remote/UK-wide",
        "salary": "£60,000 - £95,000",
        "description": "Analyse complex datasets; use AI tools for predictive analytics; support business intelligence",
        "ai_impact": "AI tools integral; driving data-driven decisions",
        "date_posted": "Feb 2025",
        "source": "[1][4][5]"
    },
    {
        "job_id": 4,
        "title": "Automation Engineer",
        "company": "Robotics Innovator",
        "location": "Birmingham",
        "salary": "£55,000 - £90,000",
        "description": "Design and implement AI-powered automation systems; robotics process automation",
        "ai_impact": "AI enables automation; high growth area",
        "date_posted": "Feb 2025",
        "source": "[4]"
    },
    {
        "job_id": 5,
        "title": "Cybersecurity Analyst",
        "company": "Financial Institution",
        "location": "London",
        "salary": "£50,000 - £85,000",
        "description": "Monitor and respond to AI-generated threat intelligence; use AI for anomaly detection",
        "ai_impact": "AI enhances threat detection and response",
        "date_posted": "Feb 2025",
        "source": "[1][5]"
    },
    {
        "job_id": 6,
        "title": "Cloud Solutions Architect",
        "company": "AWS Partner",
        "location": "Manchester/Remote",
        "salary": "£75,000 - £115,000",
        "description": "Architect cloud infrastructure with AI integration; ensure scalable AI deployments",
        "ai_impact": "AI cloud integration critical",
        "date_posted": "Feb 2025",
        "source": "[1][4][5]"
    }
]

def create_and_populate_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Create table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY,
            title TEXT,
            company TEXT,
            location TEXT,
            salary TEXT,
            description TEXT,
            ai_impact TEXT,
            date_posted TEXT,
            source TEXT
        )
    ''')
    # Insert data
    for job in jobs_data:
        c.execute('''
            INSERT OR REPLACE INTO jobs
            (job_id, title, company, location, salary, description, ai_impact, date_posted, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job["job_id"],
            job["title"],
            job["company"],
            job["location"],
            job["salary"],
            job["description"],
            job["ai_impact"],
            job["date_posted"],
            job["source"]
        ))
    conn.commit()
    conn.close()
    print(f"Inserted {len(jobs_data)} job postings into {db_path}")

if __name__ == "__main__":
    create_and_populate_db("db/data.db")
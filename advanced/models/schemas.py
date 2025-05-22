"""
Database schema definitions for the advanced DuckDB implementation.

This module contains SQL statements for creating tables, indexes, and other
database objects for the UK AI jobs analytics database.
"""

# Job postings table
JOB_POSTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS job_postings (
    job_id VARCHAR PRIMARY KEY,
    title VARCHAR NOT NULL,
    company VARCHAR NOT NULL,
    location VARCHAR NOT NULL,
    salary VARCHAR,
    salary_min FLOAT,  -- Extracted minimum salary
    salary_max FLOAT,  -- Extracted maximum salary
    salary_currency VARCHAR(3),  -- Currency code (GBP, USD, etc.)
    description TEXT,
    responsibilities TEXT,  -- Separated responsibilities section
    requirements TEXT,  -- Separated requirements section
    benefits TEXT,  -- Separated benefits section
    ai_impact FLOAT,  -- Numeric score (0-1)
    ai_impact_category VARCHAR,  -- Categorical (low, medium, high, transformative)
    remote_work BOOLEAN,  -- Whether remote work is available
    remote_percentage INTEGER,  -- Percentage of remote work allowed (0-100)
    contract_type VARCHAR,  -- Full-time, part-time, contract, etc.
    seniority_level VARCHAR,  -- Junior, mid-level, senior, etc.
    date_posted DATE NOT NULL,
    application_deadline DATE,
    source VARCHAR NOT NULL,
    source_url VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# Job skills table
JOB_SKILLS_TABLE = """
CREATE TABLE IF NOT EXISTS job_skills (
    id INTEGER PRIMARY KEY,
    job_id VARCHAR NOT NULL,
    skill_name VARCHAR NOT NULL,
    skill_category VARCHAR,  -- Technical, soft skill, tool, etc.
    is_required BOOLEAN,  -- Required vs. preferred
    experience_years INTEGER,  -- Required years of experience
    FOREIGN KEY (job_id) REFERENCES job_postings(job_id) ON DELETE CASCADE,
    UNIQUE(job_id, skill_name)
)
"""

# Companies table
COMPANIES_TABLE = """
CREATE TABLE IF NOT EXISTS companies (
    company_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    industry VARCHAR,
    company_size VARCHAR,  -- Small, medium, large, enterprise
    founded_year INTEGER,
    headquarters VARCHAR,
    website VARCHAR,
    ai_focus_level FLOAT,  -- How focused the company is on AI (0-1)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# Job history table for tracking changes
JOB_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS job_history (
    history_id INTEGER PRIMARY KEY,
    job_id VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,  -- Created, updated, removed, etc.
    field_name VARCHAR,  -- Which field was changed
    old_value TEXT,
    new_value TEXT,
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES job_postings(job_id) ON DELETE CASCADE
)
"""

# AI impact metrics table
AI_IMPACT_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS ai_impact_metrics (
    metric_id INTEGER PRIMARY KEY,
    job_id VARCHAR NOT NULL,
    automation_risk FLOAT,  -- Risk of automation (0-1)
    augmentation_potential FLOAT,  -- Potential for AI to augment the role (0-1)
    transformation_level FLOAT,  -- How much AI transforms the role (0-1)
    analysis_date DATE NOT NULL,
    analysis_method VARCHAR,  -- How the metrics were derived
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES job_postings(job_id) ON DELETE CASCADE
)
"""

# Perplexity data sources tracking table
PERPLEXITY_DATA_SOURCES_TABLE = """
CREATE TABLE IF NOT EXISTS perplexity_data_sources (
    source_id INTEGER PRIMARY KEY,
    query_text TEXT NOT NULL,
    response_id VARCHAR,
    data_retrieval_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    citation_links TEXT,
    response_summary TEXT
)
"""

# Data quality metrics table
DATA_QUALITY_METRICS_TABLE = """
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    metric_id INTEGER PRIMARY KEY,
    table_name VARCHAR NOT NULL,
    column_name VARCHAR,
    completeness FLOAT,  -- Percentage of non-null values
    accuracy FLOAT,      -- Estimated accuracy based on validation rules
    consistency FLOAT,   -- Consistency with related data
    check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# Index definitions
INDEX_DEFINITIONS = [
    "CREATE INDEX IF NOT EXISTS idx_title ON job_postings(title)",
    "CREATE INDEX IF NOT EXISTS idx_company ON job_postings(company)",
    "CREATE INDEX IF NOT EXISTS idx_location ON job_postings(location)",
    "CREATE INDEX IF NOT EXISTS idx_date_posted ON job_postings(date_posted)",
    "CREATE INDEX IF NOT EXISTS idx_ai_impact ON job_postings(ai_impact)",
    "CREATE INDEX IF NOT EXISTS idx_remote_work ON job_postings(remote_work)",
    "CREATE INDEX IF NOT EXISTS idx_seniority_level ON job_postings(seniority_level)",
    "CREATE INDEX IF NOT EXISTS idx_job_skills_job_id ON job_skills(job_id)",
    "CREATE INDEX IF NOT EXISTS idx_job_skills_skill_name ON job_skills(skill_name)",
    "CREATE INDEX IF NOT EXISTS idx_ai_impact_metrics_job_id ON ai_impact_metrics(job_id)",
    "CREATE INDEX IF NOT EXISTS idx_ai_impact_category ON job_postings(ai_impact_category)",
]

# All tables in creation order
ALL_TABLES = [
    JOB_POSTINGS_TABLE,
    JOB_SKILLS_TABLE,
    COMPANIES_TABLE,
    JOB_HISTORY_TABLE,
    AI_IMPACT_METRICS_TABLE,
    PERPLEXITY_DATA_SOURCES_TABLE,
    DATA_QUALITY_METRICS_TABLE
]

# Common queries
COMMON_QUERIES = {
    "get_job_by_id": """
        SELECT * FROM job_postings 
        WHERE job_id = ?
    """,
    
    "get_job_skills": """
        SELECT skill_name, skill_category, is_required 
        FROM job_skills 
        WHERE job_id = ?
    """,
    
    "get_job_ai_metrics": """
        SELECT automation_risk, augmentation_potential, transformation_level 
        FROM ai_impact_metrics 
        WHERE job_id = ? 
        ORDER BY analysis_date DESC 
        LIMIT 1
    """,
    
    "get_company_by_job_id": """
        SELECT c.* 
        FROM companies c 
        JOIN job_postings j ON c.name = j.company 
        WHERE j.job_id = ?
    """,
    
    "search_jobs_basic": """
        SELECT * FROM job_postings
        WHERE 1=1
        {title_filter}
        {company_filter}
        {location_filter}
        {ai_impact_filter}
        {remote_filter}
        {date_filter}
        ORDER BY date_posted DESC
        LIMIT ? OFFSET ?
    """,
    
    "count_jobs_by_category": """
        SELECT ai_impact_category, COUNT(*) as job_count
        FROM job_postings
        GROUP BY ai_impact_category
        ORDER BY job_count DESC
    """,
    
    "avg_salary_by_category": """
        SELECT ai_impact_category, 
               AVG(salary_min) as avg_min_salary,
               AVG(salary_max) as avg_max_salary
        FROM job_postings
        WHERE salary_min IS NOT NULL AND salary_max IS NOT NULL
        GROUP BY ai_impact_category
    """,
    
    "top_skills_by_ai_impact": """
        SELECT s.skill_name, s.skill_category, COUNT(*) as skill_count,
               AVG(j.ai_impact) as avg_ai_impact
        FROM job_skills s
        JOIN job_postings j ON s.job_id = j.job_id
        GROUP BY s.skill_name, s.skill_category
        ORDER BY avg_ai_impact DESC, skill_count DESC
        LIMIT ?
    """,
    
    "jobs_by_date_trend": """
        SELECT strftime('%Y-%m', date_posted) as month,
               COUNT(*) as job_count,
               AVG(ai_impact) as avg_ai_impact
        FROM job_postings
        GROUP BY month
        ORDER BY month
    """
}
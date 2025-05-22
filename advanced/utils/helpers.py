"""
Helper functions for the advanced DuckDB implementation.

This module provides utility functions used throughout the application.
"""

import os
import json
import datetime
import re
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union

import pandas as pd

from advanced.models.data_classes import (
    JobPosting, Skill, Salary, AIImpactLevel, ContractType, SeniorityLevel, SkillCategory
)
from advanced.utils.logging import setup_logging, timed_function

# Set up logging
logger = setup_logging(__name__)


@timed_function
def extract_salary_info(salary_text: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Extract salary information from text.
    
    Args:
        salary_text: Text containing salary information
        
    Returns:
        Tuple of (min_salary, max_salary, currency)
    """
    min_salary = None
    max_salary = None
    currency = "GBP"  # Default for UK jobs
    
    try:
        # Detect currency
        if '£' in salary_text:
            currency = "GBP"
        elif '$' in salary_text:
            currency = "USD"
        elif '€' in salary_text:
            currency = "EUR"
        
        # Remove currency symbols and commas
        cleaned_text = salary_text.replace('£', '').replace('$', '').replace('€', '').replace(',', '')
        
        # Extract the range if present
        if '-' in cleaned_text:
            parts = cleaned_text.split('-')
            try:
                # Extract first number
                min_value_str = re.search(r'\d+', parts[0])
                if min_value_str:
                    min_salary = float(min_value_str.group())
                
                # Extract second number
                max_value_str = re.search(r'\d+', parts[1])
                if max_value_str:
                    max_salary = float(max_value_str.group())
            except (ValueError, IndexError):
                pass
        else:
            # Check for keywords like "up to" or "from"
            if 'up to' in salary_text.lower():
                max_value_str = re.search(r'\d+', cleaned_text)
                if max_value_str:
                    max_salary = float(max_value_str.group())
                    min_salary = max_salary * 0.7  # Estimate min as 70% of max
            elif 'from' in salary_text.lower() or 'minimum' in salary_text.lower():
                min_value_str = re.search(r'\d+', cleaned_text)
                if min_value_str:
                    min_salary = float(min_value_str.group())
                    max_salary = min_salary * 1.3  # Estimate max as 130% of min
    except Exception as e:
        logger.warning(f"Failed to extract salary information from '{salary_text}': {e}")
    
    return min_salary, max_salary, currency


@timed_function
def parse_date(date_text: str) -> datetime.date:
    """
    Parse a date string in various formats.
    
    Args:
        date_text: Date string to parse
        
    Returns:
        datetime.date object
    """
    formats = ['%Y-%m-%d', '%d/%m/%Y', '%b %Y', '%B %Y', '%Y']
    
    # Try standard formats
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_text, fmt).date()
        except ValueError:
            continue
    
    # Handle special cases
    if "2025" in date_text:
        if "early" in date_text.lower():
            return datetime.date(2025, 1, 15)
        elif "mid" in date_text.lower():
            return datetime.date(2025, 6, 15)
        elif "late" in date_text.lower():
            return datetime.date(2025, 10, 15)
        else:
            return datetime.date(2025, 1, 1)
    
    # Handle "X days/weeks/months ago"
    ago_match = re.search(r'(\d+)\s+(day|week|month)s?\s+ago', date_text.lower())
    if ago_match:
        num = int(ago_match.group(1))
        unit = ago_match.group(2)
        today = datetime.date.today()
        
        if unit == 'day':
            return today - datetime.timedelta(days=num)
        elif unit == 'week':
            return today - datetime.timedelta(days=num * 7)
        elif unit == 'month':
            # Approximate months as 30 days
            return today - datetime.timedelta(days=num * 30)
    
    # Default to today
    return datetime.date.today()


@timed_function
def generate_unique_id(prefix: str = "JOB") -> str:
    """
    Generate a unique ID with the specified prefix.
    
    Args:
        prefix: Prefix for the ID (default: "JOB")
        
    Returns:
        Unique ID string
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@timed_function
def save_dataframe(df: pd.DataFrame, filename: str, directory: str = None) -> str:
    """
    Save a pandas DataFrame to a file.
    
    Args:
        df: DataFrame to save
        filename: Name of the file (without path)
        directory: Directory to save to (uses exports directory from config if None)
        
    Returns:
        Full path to the saved file
    """
    from advanced.config import ANALYTICS_CONFIG
    
    if directory is None:
        directory = ANALYTICS_CONFIG.get("export_dir")
    
    os.makedirs(directory, exist_ok=True)
    
    # Determine file format based on extension
    filepath = os.path.join(directory, filename)
    extension = os.path.splitext(filename)[1].lower()
    
    if extension == '.csv':
        df.to_csv(filepath, index=False)
    elif extension == '.xlsx':
        df.to_excel(filepath, index=False)
    elif extension == '.json':
        df.to_json(filepath, orient='records', lines=True)
    elif extension == '.html':
        df.to_html(filepath, index=False)
    elif extension == '.md':
        with open(filepath, 'w') as f:
            f.write(df.to_markdown(index=False))
    else:
        # Default to CSV
        filepath = f"{os.path.splitext(filepath)[0]}.csv"
        df.to_csv(filepath, index=False)
    
    logger.info(f"Saved DataFrame to {filepath}")
    return filepath


@timed_function
def create_sample_job_data(count: int = 5) -> List[JobPosting]:
    """
    Create sample job posting data for testing.
    
    Args:
        count: Number of job postings to create
        
    Returns:
        List of JobPosting objects
    """
    job_titles = [
        "Data Scientist", "Machine Learning Engineer", "AI Research Scientist",
        "Data Engineer", "DevOps Engineer", "Software Developer", "Frontend Developer",
        "Backend Developer", "Full Stack Developer", "Product Manager", "UX Designer"
    ]
    
    companies = [
        "TechCorp UK", "AI Solutions Ltd", "Data Insights Co", "Cloud Services UK",
        "Innovative Systems", "Digital Transformers", "Tech Innovate", "Future Tech"
    ]
    
    locations = [
        "London, UK", "Manchester, UK", "Birmingham, UK", "Edinburgh, UK",
        "Glasgow, UK", "Liverpool, UK", "Leeds, UK", "Bristol, UK",
        "Remote, UK", "London (Hybrid), UK"
    ]
    
    skills_by_category = {
        SkillCategory.TECHNICAL: [
            "Python", "Java", "JavaScript", "TypeScript", "C#", "C++",
            "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL"
        ],
        SkillCategory.AI: [
            "TensorFlow", "PyTorch", "scikit-learn", "Machine Learning",
            "Deep Learning", "NLP", "Computer Vision", "Reinforcement Learning"
        ],
        SkillCategory.FRAMEWORK: [
            "React", "Angular", "Vue", "Django", "Flask", "Spring", "Express"
        ],
        SkillCategory.TOOL: [
            "Git", "Docker", "Kubernetes", "Jenkins", "AWS", "Azure", "GCP"
        ]
    }
    
    job_postings = []
    
    import random
    
    for i in range(count):
        # Select job title based on index
        title_idx = i % len(job_titles)
        title = job_titles[title_idx]
        
        # Determine AI impact based on title
        if "AI" in title or "Machine Learning" in title or "Data Scientist" in title:
            ai_impact = random.uniform(0.7, 0.95)
        elif "Data" in title or "Engineer" in title:
            ai_impact = random.uniform(0.4, 0.7)
        else:
            ai_impact = random.uniform(0.1, 0.4)
        
        # Generate random date within the last year
        days_ago = random.randint(1, 365)
        date_posted = datetime.date.today() - datetime.timedelta(days=days_ago)
        
        # Generate random salary range
        base_salary = 40000
        if ai_impact > 0.7:
            min_salary = base_salary + random.randint(20000, 30000)
        elif ai_impact > 0.4:
            min_salary = base_salary + random.randint(10000, 20000)
        else:
            min_salary = base_salary + random.randint(0, 10000)
        
        max_salary = min_salary + random.randint(5000, 20000)
        
        # Determine if remote
        remote_work = random.choice([True, False])
        remote_percentage = random.choice([0, 30, 50, 100]) if remote_work else 0
        
        # Add location hint about remote work
        location = locations[i % len(locations)]
        if remote_work and "Remote" not in location:
            if remote_percentage == 100:
                location = f"Remote, UK"
            else:
                location = f"{location} (Hybrid)"
        
        # Generate random skills
        num_technical_skills = random.randint(1, 4)
        num_ai_skills = random.randint(0, 3) if ai_impact > 0.4 else 0
        num_framework_skills = random.randint(0, 2)
        num_tool_skills = random.randint(0, 2)
        
        skills = []
        
        for category, count, skill_list in [
            (SkillCategory.TECHNICAL, num_technical_skills, skills_by_category[SkillCategory.TECHNICAL]),
            (SkillCategory.AI, num_ai_skills, skills_by_category[SkillCategory.AI]),
            (SkillCategory.FRAMEWORK, num_framework_skills, skills_by_category[SkillCategory.FRAMEWORK]),
            (SkillCategory.TOOL, num_tool_skills, skills_by_category[SkillCategory.TOOL])
        ]:
            selected_skills = random.sample(skill_list, min(count, len(skill_list)))
            for skill_name in selected_skills:
                skills.append(Skill(
                    name=skill_name,
                    category=category,
                    is_required=random.choice([True, False]),
                    experience_years=random.randint(0, 5)
                ))
        
        # Create job posting
        job_posting = JobPosting(
            job_id=generate_unique_id(),
            title=title,
            company=companies[i % len(companies)],
            location=location,
            description=f"We are looking for a talented {title} to join our team...",
            date_posted=date_posted,
            source="sample_data",
            ai_impact=ai_impact,
            salary_text=f"£{min_salary} - £{max_salary}",
            salary=Salary(min_value=min_salary, max_value=max_salary, currency="GBP"),
            remote_work=remote_work,
            remote_percentage=remote_percentage,
            contract_type=random.choice(list(ContractType)),
            seniority_level=random.choice(list(SeniorityLevel)),
            skills=skills
        )
        
        job_postings.append(job_posting)
    
    return job_postings
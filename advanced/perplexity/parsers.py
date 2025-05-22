"""
Parsers for PerplexityAI responses in the advanced DuckDB implementation.

This module provides utilities to parse PerplexityAI responses and extract structured
job data for insertion into the database.
"""

import re
import logging
import datetime
import uuid
from typing import Dict, List, Optional, Any, Tuple
import random

import pandas as pd

from advanced.models.data_classes import (
    JobPosting, Skill, Salary, AIImpactLevel, ContractType, SeniorityLevel, SkillCategory
)
from advanced.config import AI_SKILLS, TECH_SKILLS

# Set up logging
logger = logging.getLogger(__name__)


def extract_table_data(content: str) -> Optional[pd.DataFrame]:
    """
    Extract tabular data from PerplexityAI response content.
    
    Args:
        content: The response content from PerplexityAI
        
    Returns:
        pandas DataFrame containing the extracted table data, or None if extraction fails
    """
    try:
        # Look for markdown tables
        table_pattern = r'\|(.+?)\|\s*\n\|(?:-+\|)+\s*\n((?:\|.+?\|\s*\n)*)'
        table_matches = re.findall(table_pattern, content, re.DOTALL)
        
        if table_matches:
            # Process the first table found
            headers = [h.strip() for h in table_matches[0][0].split('|') if h.strip()]
            rows = []
            
            row_data = table_matches[0][1].strip().split('\n')
            for row in row_data:
                cells = [cell.strip() for cell in row.split('|') if cell.strip()]
                if cells:
                    rows.append(cells)
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            logger.info(f"Extracted table with {len(df)} rows and {len(headers)} columns")
            return df
        else:
            logger.warning("No table found in response content")
            return None
    except Exception as e:
        logger.error(f"Failed to extract table data: {e}")
        return None


def estimate_ai_impact(job_title: str, description: str) -> float:
    """
    Estimate AI impact score based on job title and description.
    
    Args:
        job_title: The job title
        description: The job description
        
    Returns:
        Estimated AI impact score (0-1)
    """
    impact_score = 0.0
    
    # High impact job titles
    high_impact_keywords = [
        'ai', 'artificial intelligence', 'machine learning', 'ml engineer', 
        'data scientist', 'nlp', 'computer vision'
    ]
    
    # Medium impact job titles
    medium_impact_keywords = [
        'data engineer', 'cloud architect', 'backend developer',
        'analytics', 'big data'
    ]
    
    # Check title for keywords
    title_lower = job_title.lower()
    for keyword in high_impact_keywords:
        if keyword in title_lower:
            impact_score += 0.4
            break
    
    for keyword in medium_impact_keywords:
        if keyword in title_lower:
            impact_score += 0.2
            break
    
    # Check description for AI skills
    description_lower = description.lower()
    ai_skill_count = 0
    for skill in AI_SKILLS:
        if skill.lower() in description_lower:
            ai_skill_count += 1
    
    # Adjust score based on AI skill density
    if ai_skill_count > 5:
        impact_score += 0.5
    elif ai_skill_count > 2:
        impact_score += 0.3
    elif ai_skill_count > 0:
        impact_score += 0.1
    
    # Ensure score is in range 0-1
    impact_score = min(max(impact_score, 0.0), 1.0)
    
    return impact_score


def extract_salary_info(salary_text: str) -> Salary:
    """
    Extract structured salary information from salary text.
    
    Args:
        salary_text: The salary information as a string (e.g., "£60,000 - £80,000")
        
    Returns:
        Salary object with extracted min and max values
    """
    # Default values
    min_value = None
    max_value = None
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
                    min_value = float(min_value_str.group())
                
                # Extract second number
                max_value_str = re.search(r'\d+', parts[1])
                if max_value_str:
                    max_value = float(max_value_str.group())
            except (ValueError, IndexError):
                pass
        else:
            # Check for keywords like "up to" or "from"
            if 'up to' in salary_text.lower():
                max_value_str = re.search(r'\d+', cleaned_text)
                if max_value_str:
                    max_value = float(max_value_str.group())
                    min_value = max_value * 0.7  # Estimate min as 70% of max
            elif 'from' in salary_text.lower() or 'minimum' in salary_text.lower():
                min_value_str = re.search(r'\d+', cleaned_text)
                if min_value_str:
                    min_value = float(min_value_str.group())
                    max_value = min_value * 1.3  # Estimate max as 130% of min
    except Exception as e:
        logger.warning(f"Failed to extract salary information from '{salary_text}': {e}")
    
    return Salary(min_value=min_value, max_value=max_value, currency=currency)


def extract_skills_from_description(description: str) -> List[Skill]:
    """
    Extract skills from job description using keyword matching.
    
    Args:
        description: Job description text
        
    Returns:
        List of Skill objects
    """
    skills = []
    description_lower = description.lower()
    
    # Check for technical skills
    for skill_name in TECH_SKILLS:
        if skill_name.lower() in description_lower:
            is_required = "required" in description_lower[
                max(0, description_lower.find(skill_name.lower()) - 50):
                min(len(description_lower), description_lower.find(skill_name.lower()) + 50)
            ]
            skills.append(Skill(
                name=skill_name,
                category=SkillCategory.TECHNICAL,
                is_required=is_required
            ))
    
    # Check for AI skills
    for skill_name in AI_SKILLS:
        if skill_name.lower() in description_lower:
            is_required = "required" in description_lower[
                max(0, description_lower.find(skill_name.lower()) - 50):
                min(len(description_lower), description_lower.find(skill_name.lower()) + 50)
            ]
            skills.append(Skill(
                name=skill_name,
                category=SkillCategory.AI,
                is_required=is_required
            ))
    
    return skills


def determine_seniority(title: str, description: str) -> Optional[SeniorityLevel]:
    """
    Determine the seniority level from job title and description.
    
    Args:
        title: Job title
        description: Job description
        
    Returns:
        SeniorityLevel enum value or None if not determinable
    """
    title_lower = title.lower()
    description_lower = description.lower()
    combined_text = title_lower + " " + description_lower
    
    # Check for explicit seniority in title
    if "senior" in title_lower or "sr." in title_lower:
        return SeniorityLevel.SENIOR
    elif "junior" in title_lower or "jr." in title_lower:
        return SeniorityLevel.JUNIOR
    elif "lead" in title_lower:
        return SeniorityLevel.LEAD
    elif "manager" in title_lower:
        return SeniorityLevel.MANAGER
    elif "director" in title_lower:
        return SeniorityLevel.DIRECTOR
    elif "entry" in title_lower or "graduate" in title_lower:
        return SeniorityLevel.ENTRY
    
    # Check for seniority indications in description
    if "senior" in combined_text or "experienced" in combined_text:
        return SeniorityLevel.SENIOR
    elif "junior" in combined_text or "entry level" in combined_text:
        return SeniorityLevel.JUNIOR
    elif "mid-level" in combined_text or "intermediate" in combined_text:
        return SeniorityLevel.MID_LEVEL
    
    # Default to mid-level if not determinable
    return None


def determine_contract_type(description: str) -> Optional[ContractType]:
    """
    Determine the contract type from job description.
    
    Args:
        description: Job description
        
    Returns:
        ContractType enum value or None if not determinable
    """
    description_lower = description.lower()
    
    if "full-time" in description_lower or "full time" in description_lower:
        return ContractType.FULL_TIME
    elif "part-time" in description_lower or "part time" in description_lower:
        return ContractType.PART_TIME
    elif "contract" in description_lower and ("position" in description_lower or "role" in description_lower):
        return ContractType.CONTRACT
    elif "freelance" in description_lower:
        return ContractType.FREELANCE
    elif "internship" in description_lower:
        return ContractType.INTERNSHIP
    elif "temporary" in description_lower:
        return ContractType.TEMPORARY
    
    # Default to full-time if not specified
    return ContractType.FULL_TIME


def perplexity_to_job_postings(content: str, source: str = "perplexity") -> List[JobPosting]:
    """
    Convert PerplexityAI response content to JobPosting objects.
    
    Args:
        content: The response content from PerplexityAI
        source: Source identifier for the data
        
    Returns:
        List of JobPosting objects
    """
    job_postings = []
    
    try:
        # Extract table data
        df = extract_table_data(content)
        
        if df is None or len(df) == 0:
            logger.warning("No table data found in response")
            return job_postings
        
        # Map expected columns to actual columns
        column_mapping = {}
        expected_columns = [
            'Job Title', 'Title', 'Salary Range', 'Salary', 'Location', 
            'Job Description', 'Description', 'Impact Metrics', 'AI Impact',
            'Posting Date', 'Date Posted', 'Data Source', 'Source', 'Company'
        ]
        
        for expected in expected_columns:
            for actual in df.columns:
                if expected.lower() in actual.lower():
                    column_mapping[expected] = actual
                    break
        
        # Process each row
        for _, row in df.iterrows():
            try:
                # Extract basic information
                title = row.get(column_mapping.get('Job Title', column_mapping.get('Title', None)))
                company = row.get(column_mapping.get('Company', None), "Unknown Company")
                location = row.get(column_mapping.get('Location', None), "UK")
                description = row.get(column_mapping.get('Job Description', column_mapping.get('Description', None)), "")
                salary_text = row.get(column_mapping.get('Salary Range', column_mapping.get('Salary', None)), "")
                impact_text = row.get(column_mapping.get('Impact Metrics', column_mapping.get('AI Impact', None)), "")
                date_text = row.get(column_mapping.get('Posting Date', column_mapping.get('Date Posted', None)), "")
                data_source = row.get(column_mapping.get('Data Source', column_mapping.get('Source', None)), source)
                
                # Skip if title is missing
                if not title:
                    continue
                
                # Generate job ID
                job_id = f"JOB-{uuid.uuid4().hex[:8]}"
                
                # Process posting date
                try:
                    date_posted = parse_date(date_text)
                except:
                    date_posted = datetime.date.today()
                
                # Estimate AI impact if not provided
                try:
                    ai_impact = float(impact_text) if impact_text and impact_text.replace('.', '', 1).isdigit() else None
                except:
                    ai_impact = None
                
                if ai_impact is None:
                    ai_impact = estimate_ai_impact(title, description)
                
                # Extract salary information
                salary = extract_salary_info(salary_text)
                
                # Extract skills
                skills = extract_skills_from_description(description)
                
                # Determine seniority and contract type
                seniority_level = determine_seniority(title, description)
                contract_type = determine_contract_type(description)
                
                # Determine remote work
                remote_work = "remote" in location.lower() or "remote" in description.lower()
                remote_percentage = 100 if "fully remote" in (location + description).lower() else 50 if remote_work else 0
                
                # Create job posting
                job_posting = JobPosting(
                    job_id=job_id,
                    title=title,
                    company=company,
                    location=location,
                    description=description,
                    date_posted=date_posted,
                    source=data_source,
                    ai_impact=ai_impact,
                    salary_text=salary_text,
                    salary=salary,
                    remote_work=remote_work,
                    remote_percentage=remote_percentage,
                    contract_type=contract_type,
                    seniority_level=seniority_level,
                    skills=skills
                )
                
                job_postings.append(job_posting)
                
            except Exception as e:
                logger.error(f"Failed to process row: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Failed to convert PerplexityAI response to job postings: {e}")
    
    logger.info(f"Extracted {len(job_postings)} job postings from PerplexityAI response")
    return job_postings


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
"""
Data processor module for PerplexityAI MCP integration.

This module provides functionality to process PerplexityAI responses,
extract structured job data, normalize and clean the data, calculate
AI impact metrics, and prepare the data for insertion into the DuckDB database.
"""

import re
import json
import logging
import datetime
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import asdict

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from advanced.models.data_classes import (
    JobPosting, Skill, Salary, AIImpactLevel, ContractType, 
    SeniorityLevel, SkillCategory, PerplexityResponse, JobMetrics
)
from advanced.config import AI_SKILLS, TECH_SKILLS
from advanced.utils.helpers import extract_salary_info, parse_date, generate_unique_id
from advanced.utils.logging import setup_logging, timed_function

# Set up logging
logger = setup_logging(__name__)


class DataProcessor:
    """
    Data processor for PerplexityAI responses.
    
    This class provides methods to process and transform PerplexityAI responses
    into structured job data suitable for database insertion.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        # Initialize skill lookups for faster processing
        self.ai_skills_set = {skill.lower() for skill in AI_SKILLS}
        self.tech_skills_set = {skill.lower() for skill in TECH_SKILLS}
        
        # Compile regex patterns for efficiency
        self.table_pattern = re.compile(r'\|(.+?)\|\s*\n\|(?:-+\|)+\s*\n((?:\|.+?\|\s*\n)*)', re.DOTALL)
        self.salary_pattern = re.compile(r'\d+\.?\d*')
        self.date_ago_pattern = re.compile(r'(\d+)\s+(day|week|month)s?\s+ago')
        
        # Location normalization mappings
        self.location_aliases = {
            "london": "London, UK",
            "manchester": "Manchester, UK",
            "birmingham": "Birmingham, UK",
            "leeds": "Leeds, UK",
            "glasgow": "Glasgow, UK",
            "liverpool": "Liverpool, UK",
            "edinburgh": "Edinburgh, UK",
            "bristol": "Bristol, UK",
            "remote uk": "Remote, UK",
            "remote": "Remote, UK",
            "uk remote": "Remote, UK"
        }
        
        # Initialize vectorizer for text analysis
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    @timed_function
    def process_perplexity_response(self, response: PerplexityResponse) -> List[JobPosting]:
        """
        Process a PerplexityAI response and extract job postings.
        
        Args:
            response: PerplexityResponse object containing the response data
            
        Returns:
            List of JobPosting objects extracted from the response
        """
        logger.info(f"Processing PerplexityAI response with ID: {response.response_id}")
        
        # First try to extract structured data (tables, JSON, etc.)
        job_data = self.extract_structured_data(response.content)
        
        if not job_data:
            # Fall back to extracting unstructured data
            logger.info("No structured data found, trying to extract unstructured data")
            job_data = self.extract_unstructured_data(response.content)
        
        # Transform extracted data into JobPosting objects
        job_postings = []
        for data in job_data:
            try:
                job_posting = self.transform_to_job_posting(data, source=response.response_id)
                if job_posting:
                    # Calculate additional metrics for the job posting
                    job_posting = self.calculate_ai_impact_metrics(job_posting)
                    job_postings.append(job_posting)
            except Exception as e:
                logger.error(f"Failed to transform job data: {e}")
                continue
        
        logger.info(f"Extracted {len(job_postings)} job postings from response")
        return job_postings
    
    @timed_function
    def extract_structured_data(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract structured data from the response content.
        
        Args:
            content: Response content as a string
            
        Returns:
            List of dictionaries containing job data
        """
        # Try to extract data from tables first
        table_data = self.extract_table_data(content)
        if table_data and len(table_data) > 0:
            return table_data
        
        # Try to extract JSON data
        json_data = self.extract_json_data(content)
        if json_data and len(json_data) > 0:
            return json_data
        
        # Try to extract key-value pairs
        kv_data = self.extract_key_value_data(content)
        if kv_data and len(kv_data) > 0:
            return kv_data
        
        return []
    
    @timed_function
    def extract_table_data(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract tabular data from the response content.
        
        Args:
            content: Response content as a string
            
        Returns:
            List of dictionaries containing job data extracted from tables
        """
        table_matches = self.table_pattern.findall(content)
        if not table_matches:
            return []
        
        job_data = []
        
        for header_row, data_rows in table_matches:
            try:
                # Process headers
                headers = [h.strip() for h in header_row.split('|') if h.strip()]
                
                # Process rows
                rows_data = []
                for row in data_rows.strip().split('\n'):
                    cells = [cell.strip() for cell in row.split('|') if cell.strip()]
                    if cells and len(cells) == len(headers):
                        rows_data.append(cells)
                
                # Create dictionaries for each row
                for row_cells in rows_data:
                    job_data.append(dict(zip(headers, row_cells)))
                
                logger.info(f"Extracted {len(rows_data)} rows from table with headers: {headers}")
            except Exception as e:
                logger.error(f"Failed to extract table data: {e}")
        
        return job_data
    
    @timed_function
    def extract_json_data(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract JSON data from the response content.
        
        Args:
            content: Response content as a string
            
        Returns:
            List of dictionaries containing job data extracted from JSON
        """
        # Try to find JSON arrays in the text
        json_pattern = r'(\[\s*\{.*?\}\s*\])'
        json_matches = re.findall(json_pattern, content, re.DOTALL)
        
        job_data = []
        
        for json_str in json_matches:
            try:
                parsed_data = json.loads(json_str)
                if isinstance(parsed_data, list) and all(isinstance(item, dict) for item in parsed_data):
                    job_data.extend(parsed_data)
                    logger.info(f"Extracted {len(parsed_data)} job entries from JSON data")
            except json.JSONDecodeError:
                pass
        
        # If no JSON arrays found, try to find JSON objects
        if not job_data:
            json_obj_pattern = r'(\{\s*".*?"\s*:.*?\})'
            json_obj_matches = re.findall(json_obj_pattern, content, re.DOTALL)
            
            for json_str in json_obj_matches:
                try:
                    parsed_data = json.loads(json_str)
                    if isinstance(parsed_data, dict):
                        job_data.append(parsed_data)
                        logger.info("Extracted job entry from JSON object")
                except json.JSONDecodeError:
                    pass
        
        return job_data
    
    @timed_function
    def extract_key_value_data(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract key-value pairs from the response content.
        
        Args:
            content: Response content as a string
            
        Returns:
            List of dictionaries containing job data extracted from key-value pairs
        """
        job_data = []
        current_job = {}
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_job:
                    job_data.append(current_job)
                    current_job = {}
                continue
            
            # Check for key-value separator
            if ":" in line:
                key, value = line.split(":", 1)
                current_job[key.strip()] = value.strip()
            
            # Check for section headers (could indicate a new job entry)
            elif line.isupper() or (line.startswith("# ") or line.startswith("## ")):
                if current_job:
                    job_data.append(current_job)
                    current_job = {}
                current_job["title"] = line.strip("# ")
            
            # Try to add content to the last field if no separator
            elif current_job and len(current_job) > 0:
                last_key = list(current_job.keys())[-1]
                current_job[last_key] += " " + line
        
        # Add the last job if not empty
        if current_job:
            job_data.append(current_job)
        
        logger.info(f"Extracted {len(job_data)} job entries from key-value pairs")
        return job_data
    
    @timed_function
    def extract_unstructured_data(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract job data from unstructured text.
        
        Args:
            content: Response content as a string
            
        Returns:
            List of dictionaries containing job data
        """
        job_data = []
        
        # Split content into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_job = {}
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if paragraph looks like a job title
            if len(paragraph) < 100 and any(title_keyword in paragraph.lower() for title_keyword in ["engineer", "developer", "scientist", "analyst", "manager"]):
                # Save previous job data if exists
                if current_job and "title" in current_job:
                    job_data.append(current_job)
                
                # Start new job entry
                current_job = {"title": paragraph}
            
            # Try to categorize paragraph content based on keywords
            elif "company" in paragraph.lower() or "organisation" in paragraph.lower() or "organization" in paragraph.lower():
                current_job["company"] = paragraph
            elif "location" in paragraph.lower():
                current_job["location"] = paragraph
            elif "salary" in paragraph.lower() or "£" in paragraph or "$" in paragraph:
                current_job["salary"] = paragraph
            elif "skills" in paragraph.lower() or "requirements" in paragraph.lower():
                current_job["requirements"] = paragraph
            elif "description" in paragraph.lower() or "responsibilities" in paragraph.lower():
                current_job["description"] = paragraph
            elif current_job:
                # If we can't categorize, append to description if job exists
                if "description" in current_job:
                    current_job["description"] += "\n\n" + paragraph
                else:
                    current_job["description"] = paragraph
        
        # Add the last job if not empty
        if current_job and "title" in current_job:
            job_data.append(current_job)
        
        logger.info(f"Extracted {len(job_data)} job entries from unstructured text")
        return job_data
    
    @timed_function
    def transform_to_job_posting(self, job_data: Dict[str, Any], source: str = "perplexity") -> Optional[JobPosting]:
        """
        Transform extracted job data into a JobPosting object.
        
        Args:
            job_data: Dictionary containing job data
            source: Source identifier for the data
            
        Returns:
            JobPosting object or None if transformation fails
        """
        try:
            # Map field names with fallbacks
            field_mappings = {
                "title": ["title", "job_title", "position", "role"],
                "company": ["company", "company_name", "employer", "organization"],
                "location": ["location", "city", "place", "area"],
                "description": ["description", "job_description", "overview", "about"],
                "salary": ["salary", "salary_range", "compensation", "pay"],
                "date_posted": ["date_posted", "posting_date", "posted_on", "date"],
                "ai_impact": ["ai_impact", "ai_impact_score", "impact"],
                "requirements": ["requirements", "qualifications", "skills_required"],
                "responsibilities": ["responsibilities", "duties", "key_responsibilities"],
                "benefits": ["benefits", "perks", "advantages"],
                "contract_type": ["contract_type", "employment_type", "job_type"],
                "seniority": ["seniority", "level", "experience_level"],
                "remote": ["remote", "remote_work", "work_from_home", "wfh"]
            }
            
            # Extract fields with fallbacks
            extracted_fields = {}
            for field, aliases in field_mappings.items():
                for alias in aliases:
                    if alias in job_data:
                        extracted_fields[field] = job_data[alias]
                        break
            
            # Required fields with defaults
            job_id = job_data.get("job_id", generate_unique_id())
            title = extracted_fields.get("title", "Unknown Position")
            company = extracted_fields.get("company", "Unknown Company")
            location = self.normalize_location(extracted_fields.get("location", "UK"))
            description = extracted_fields.get("description", "No description provided")
            
            # Process date posted
            date_text = extracted_fields.get("date_posted", None)
            date_posted = parse_date(date_text) if date_text else datetime.date.today()
            
            # Process salary information
            salary_text = extracted_fields.get("salary", "")
            if salary_text:
                min_value, max_value, currency = extract_salary_info(salary_text)
                salary = Salary(min_value=min_value, max_value=max_value, currency=currency)
            else:
                salary = None
            
            # Process remote work information
            remote_text = extracted_fields.get("remote", "").lower()
            remote_work = (
                "remote" in location.lower() or 
                "remote" in description.lower() or
                remote_text in ["yes", "true", "1", "remote"]
            )
            
            remote_percentage = 0
            if remote_work:
                if "fully remote" in (location + description).lower():
                    remote_percentage = 100
                elif "hybrid" in (location + description).lower():
                    remote_percentage = 50
                else:
                    remote_percentage = 50  # Default for remote work with unspecified percentage
            
            # Process AI impact
            ai_impact_text = extracted_fields.get("ai_impact", "")
            try:
                ai_impact = float(ai_impact_text) if ai_impact_text and ai_impact_text.replace('.', '', 1).isdigit() else None
            except (ValueError, TypeError):
                # Handle text values like "high", "medium", etc.
                impact_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "transformative": 0.9}
                ai_impact = impact_map.get(ai_impact_text.lower() if isinstance(ai_impact_text, str) else "", None)
            
            if ai_impact is None:
                # Calculate AI impact based on title and description
                ai_impact = self.estimate_ai_impact(title, description)
            
            # Extract skills
            skills = self.extract_skills(description, extracted_fields.get("requirements", ""))
            
            # Determine contract type
            contract_type_text = extracted_fields.get("contract_type", "").lower()
            contract_type = self.determine_contract_type(contract_type_text, description)
            
            # Determine seniority level
            seniority_text = extracted_fields.get("seniority", "").lower()
            seniority_level = self.determine_seniority_level(seniority_text, title, description)
            
            # Create job posting
            job_posting = JobPosting(
                job_id=job_id,
                title=title,
                company=company,
                location=location,
                description=description,
                date_posted=date_posted,
                source=source,
                ai_impact=ai_impact,
                salary_text=salary_text,
                salary=salary,
                responsibilities=extracted_fields.get("responsibilities", None),
                requirements=extracted_fields.get("requirements", None),
                benefits=extracted_fields.get("benefits", None),
                remote_work=remote_work,
                remote_percentage=remote_percentage,
                contract_type=contract_type,
                seniority_level=seniority_level,
                skills=skills
            )
            
            return job_posting
            
        except Exception as e:
            logger.error(f"Failed to transform job data: {e}")
            return None
    
    @timed_function
    def normalize_location(self, location_text: str) -> str:
        """
        Normalize location information.
        
        Args:
            location_text: Raw location text
            
        Returns:
            Normalized location string
        """
        if not location_text:
            return "UK"
        
        location_lower = location_text.lower()
        
        # Check for direct matches in aliases
        for alias, normalized in self.location_aliases.items():
            if alias in location_lower:
                return normalized
        
        # Check for cities without "UK" suffix
        uk_cities = ["london", "manchester", "birmingham", "leeds", "glasgow", 
                    "liverpool", "edinburgh", "bristol", "sheffield", "cardiff"]
        
        for city in uk_cities:
            if city in location_lower and "uk" not in location_lower:
                return f"{city.title()}, UK"
        
        # Check for remote work mentions
        if "remote" in location_lower:
            if "hybrid" in location_lower or "partial" in location_lower:
                if any(city in location_lower for city in uk_cities):
                    # Extract city for hybrid work
                    for city in uk_cities:
                        if city in location_lower:
                            return f"{city.title()}, UK (Hybrid)"
                return "UK (Hybrid)"
            else:
                return "Remote, UK"
        
        # Default to UK if not already specified
        if "uk" not in location_lower and "united kingdom" not in location_lower:
            return f"{location_text}, UK"
        
        return location_text
    
    @timed_function
    def calculate_ai_impact_metrics(self, job_posting: JobPosting) -> JobPosting:
        """
        Calculate additional AI impact metrics for a job posting.
        
        Args:
            job_posting: JobPosting object to calculate metrics for
            
        Returns:
            JobPosting object with updated metrics
        """
        # Initialize metrics with default values
        automation_risk = 0.0
        augmentation_potential = 0.0
        transformation_level = 0.0
        
        # Calculate automation risk based on job role and skills
        title_lower = job_posting.title.lower()
        desc_lower = job_posting.description.lower()
        
        # Roles at higher risk of automation
        high_automation_risk_keywords = [
            'data entry', 'manual testing', 'support', 'helpdesk',
            'administrative', 'clerical', 'basic reporting'
        ]
        
        for keyword in high_automation_risk_keywords:
            if keyword in title_lower or keyword in desc_lower:
                automation_risk += 0.2
        
        # Skills that increase automation risk
        automatable_skills = [
            'spreadsheet', 'data entry', 'manual testing', 'basic reporting'
        ]
        
        for skill in automatable_skills:
            if skill in desc_lower:
                automation_risk += 0.1
        
        # Calculate augmentation potential
        augmentation_potential = job_posting.ai_impact * 0.8  # Base on AI impact
        
        # Adjust based on skills that are enhanced by AI
        ai_augmentable_skills = [
            'data analysis', 'research', 'content creation', 'design',
            'customer service', 'decision making', 'coding'
        ]
        
        for skill in ai_augmentable_skills:
            if skill in desc_lower:
                augmentation_potential += 0.05
        
        # Calculate transformation level based on role innovation potential
        if job_posting.ai_impact > 0.7:
            transformation_level = 0.8
        elif job_posting.ai_impact > 0.4:
            transformation_level = 0.5
        else:
            transformation_level = 0.3
        
        # Adjust based on innovation keywords
        innovation_keywords = [
            'innovate', 'transform', 'disrupt', 'pioneer', 'revolutionary',
            'cutting-edge', 'state-of-the-art', 'breakthrough'
        ]
        
        for keyword in innovation_keywords:
            if keyword in desc_lower:
                transformation_level += 0.05
        
        # Ensure all metrics are in range 0-1
        automation_risk = min(max(automation_risk, 0.0), 1.0)
        augmentation_potential = min(max(augmentation_potential, 0.0), 1.0)
        transformation_level = min(max(transformation_level, 0.0), 1.0)
        
        # Calculate required skills count
        required_skills_count = sum(1 for skill in job_posting.skills if skill.is_required)
        
        # Calculate posting recency in days
        posting_recency = (datetime.date.today() - job_posting.date_posted).days
        
        # Create JobMetrics object
        metrics = JobMetrics(
            ai_impact_score=job_posting.ai_impact,
            ai_impact_category=AIImpactLevel(job_posting.ai_impact_category),
            salary=job_posting.salary,
            remote_percentage=job_posting.remote_percentage,
            required_skills_count=required_skills_count,
            posting_recency=posting_recency,
            automation_risk=automation_risk,
            augmentation_potential=augmentation_potential,
            transformation_level=transformation_level
        )
        
        job_posting.ai_metrics = metrics
        return job_posting
    
    @timed_function
    def prepare_for_duckdb_insertion(self, job_postings: List[JobPosting]) -> pd.DataFrame:
        """
        Prepare job postings for insertion into DuckDB.
        
        Args:
            job_postings: List of JobPosting objects
            
        Returns:
            DataFrame ready for DuckDB insertion
        """
        # Convert job postings to dictionaries
        job_dicts = []
        for job in job_postings:
            job_dict = job.to_dict()
            
            # Add metrics if available
            if job.metrics:
                job_dict.update({
                    "automation_risk": job.metrics.automation_risk,
                    "augmentation_potential": job.metrics.augmentation_potential,
                    "transformation_level": job.metrics.transformation_level,
                    "required_skills_count": job.metrics.required_skills_count,
                    "posting_recency": job.metrics.posting_recency
                })
            
            job_dicts.append(job_dict)
        
        # Convert to DataFrame
        df = pd.DataFrame(job_dicts)
        
        # Ensure date columns are properly formatted
        date_columns = ['date_posted', 'application_deadline']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.date
        
        logger.info(f"Prepared {len(df)} job postings for DuckDB insertion")
        return df
    
    @timed_function
    def extract_skills_dataframe(self, job_postings: List[JobPosting]) -> pd.DataFrame:
        """
        Extract skills data from job postings for insertion into DuckDB.
        
        Args:
            job_postings: List of JobPosting objects
            
        Returns:
            DataFrame of skills data ready for DuckDB insertion
        """
        skills_data = []
        
        for job in job_postings:
            for skill in job.skills:
                skills_data.append({
                    "job_id": job.job_id,
                    "skill_name": skill.name,
                    "skill_category": skill.category.value,
                    "is_required": skill.is_required,
                    "experience_years": skill.experience_years
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(skills_data)
        logger.info(f"Extracted {len(df)} skills records for DuckDB insertion")
        return df
    
    @timed_function
    def batch_process_responses(self, responses: List[PerplexityResponse]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process multiple PerplexityAI responses in batch.
        
        Args:
            responses: List of PerplexityResponse objects
            
        Returns:
            Tuple of (jobs DataFrame, skills DataFrame) for DuckDB insertion
        """
        all_job_postings = []
        
        for response in responses:
            job_postings = self.process_perplexity_response(response)
            all_job_postings.extend(job_postings)
        
        # Prepare DataFrames for DuckDB insertion
        jobs_df = self.prepare_for_duckdb_insertion(all_job_postings)
        skills_df = self.extract_skills_dataframe(all_job_postings)
        
        # Update job posting with metrics
        job_posting.ai_metrics = metrics
        
        return job_posting
    
    @timed_function
    def extract_skills(self, description: str, requirements: Optional[str] = None) -> List[Skill]:
        """
        Extract skills from job description and requirements.
        
        Args:
            description: Job description text
            requirements: Job requirements text
            
        Returns:
            List of Skill objects
        """
        skills = []
        # Combine description and requirements for skill extraction
        combined_text = (description + " " + (requirements or "")).lower()
        
        # Extract AI skills
        for skill_name in AI_SKILLS:
            if skill_name.lower() in combined_text:
                is_required = "required" in combined_text[
                    max(0, combined_text.find(skill_name.lower()) - 50):
                    min(len(combined_text), combined_text.find(skill_name.lower()) + 50)
                ]
                
                experience_years = self.extract_experience_years(combined_text, skill_name)
                
                skills.append(Skill(
                    name=skill_name,
                    category=SkillCategory.AI,
                    is_required=is_required,
                    experience_years=experience_years
                ))
        
        # Extract technical skills
        for skill_name in TECH_SKILLS:
            if skill_name.lower() in combined_text:
                is_required = "required" in combined_text[
                    max(0, combined_text.find(skill_name.lower()) - 50):
                    min(len(combined_text), combined_text.find(skill_name.lower()) + 50)
                ]
                
                experience_years = self.extract_experience_years(combined_text, skill_name)
                
                # Determine the category
                category = SkillCategory.TECHNICAL
                if skill_name.lower() in ["aws", "azure", "gcp", "kubernetes", "docker"]:
                    category = SkillCategory.PLATFORM
                elif skill_name.lower() in ["react", "angular", "vue", "django", "flask", "spring"]:
                    category = SkillCategory.FRAMEWORK
                elif skill_name.lower() in ["python", "java", "javascript", "typescript", "c#", "c++"]:
                    category = SkillCategory.LANGUAGE
                
                skills.append(Skill(
                    name=skill_name,
                    category=category,
                    is_required=is_required,
                    experience_years=experience_years
                ))
        
        return skills
    
    @timed_function
    def extract_experience_years(self, text: str, skill_name: str) -> Optional[int]:
        """
        Extract years of experience required for a skill.
        
        Args:
            text: Text to search in
            skill_name: Name of the skill
            
        Returns:
            Number of years of experience required or None if not specified
        """
        # Search for patterns like "X years of experience with [skill]"
        patterns = [
            rf'(\d+)[\+\s]*years?[\s\w]*experience[\s\w]*{skill_name}',
            rf'{skill_name}[\s\w]*(\d+)[\+\s]*years?[\s\w]*experience',
            rf'experience[\s\w]*{skill_name}[\s\w]*(\d+)[\+\s]*years?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    pass
        
        return None
    
    @timed_function
    def estimate_ai_impact(self, job_title: str, description: str) -> float:
        """
        Estimate AI impact score based on job title and description.
        
        Args:
            job_title: The job title
            description: The job description
            
        Returns:
            Estimated AI impact score (0-1)
        """
        impact_score = 0.0
        
        # Initialize title and description for analysis
        title_lower = job_title.lower()
        description_lower = description.lower()
        
        # High impact job titles
        high_impact_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'ml engineer', 
            'data scientist', 'nlp', 'computer vision', 'deep learning'
        ]
        
        # Medium impact job titles
        medium_impact_keywords = [
            'data engineer', 'cloud architect', 'backend developer',
            'analytics', 'big data', 'software engineer', 'devops'
        ]
        
        # Check title for keywords
        for keyword in high_impact_keywords:
            if keyword in title_lower:
                impact_score += 0.4
                break
        
        for keyword in medium_impact_keywords:
            if keyword in title_lower:
                impact_score += 0.2
                break
        
        # Check description for AI skills
        ai_skill_count = 0
        for skill in self.ai_skills_set:
            if skill in description_lower:
                ai_skill_count += 1
        
        # Adjust score based on AI skill density
        if ai_skill_count > 5:
            impact_score += 0.5
        elif ai_skill_count > 2:
            impact_score += 0.3
        elif ai_skill_count > 0:
            impact_score += 0.1
        
        # Check for transformative keywords in description
        transformative_keywords = ['transform', 'revolutionize', 'cutting-edge', 'state-of-the-art', 'innovative']
        for keyword in transformative_keywords:
            if keyword in description_lower:
                impact_score += 0.05
        
        # Ensure score is in range 0-1
        impact_score = min(max(impact_score, 0.0), 1.0)
        
        return impact_score
    
    @timed_function
    def determine_contract_type(self, contract_type_text: str, description: str) -> ContractType:
        """
        Determine the contract type from text.
        
        Args:
            contract_type_text: Contract type text if available
            description: Job description text
            
        Returns:
            ContractType enum value
        """
        description_lower = description.lower()
        
        # Check for explicit contract type
        if contract_type_text:
            if "full" in contract_type_text and "time" in contract_type_text:
                return ContractType.FULL_TIME
            elif "part" in contract_type_text and "time" in contract_type_text:
                return ContractType.PART_TIME
            elif "contract" in contract_type_text:
                return ContractType.CONTRACT
            elif "freelance" in contract_type_text:
                return ContractType.FREELANCE
            elif "intern" in contract_type_text:
                return ContractType.INTERNSHIP
            elif "temp" in contract_type_text:
                return ContractType.TEMPORARY
        
        # Check description for contract type indicators
        if "full-time" in description_lower or "full time" in description_lower:
            return ContractType.FULL_TIME
        elif "part-time" in description_lower or "part time" in description_lower:
            return ContractType.PART_TIME
        elif "contract" in description_lower and ("position" in description_lower or "role" in description_lower):
            return ContractType.CONTRACT
        elif "freelance" in description_lower:
            return ContractType.FREELANCE
        elif "internship" in description_lower or "intern " in description_lower:
            return ContractType.INTERNSHIP
        elif "temporary" in description_lower or "temp " in description_lower:
            return ContractType.TEMPORARY
        
        # Default to full-time if not specified
        return ContractType.FULL_TIME
    
    @timed_function
    def determine_seniority_level(self, seniority_text: str, title: str, description: str) -> Optional[SeniorityLevel]:
        """
        Determine the seniority level from text.
        
        Args:
            seniority_text: Seniority text if available
            title: Job title
            description: Job description
            
        Returns:
            SeniorityLevel enum value or None if not determinable
        """
        title_lower = title.lower()
        description_lower = description.lower()
        combined_text = title_lower + " " + description_lower
        
        # Check for explicit seniority in provided text
        if seniority_text:
            if "senior" in seniority_text or "sr" in seniority_text:
                return SeniorityLevel.SENIOR
            elif "junior" in seniority_text or "jr" in seniority_text:
                return SeniorityLevel.JUNIOR
            elif "mid" in seniority_text:
                return SeniorityLevel.MID_LEVEL
            elif "lead" in seniority_text:
                return SeniorityLevel.LEAD
            elif "manager" in seniority_text:
                return SeniorityLevel.MANAGER
            elif "director" in seniority_text:
                return SeniorityLevel.DIRECTOR
            elif "executive" in seniority_text or "c-level" in seniority_text:
                return SeniorityLevel.EXECUTIVE
            elif "entry" in seniority_text or "graduate" in seniority_text:
                return SeniorityLevel.ENTRY
        
        # Check for explicit seniority in title
        if "senior" in title_lower or "sr." in title_lower or "sr " in title_lower:
            return SeniorityLevel.SENIOR
        elif "junior" in title_lower or "jr." in title_lower or "jr " in title_lower:
            return SeniorityLevel.JUNIOR
        elif "lead" in title_lower:
            return SeniorityLevel.LEAD
        elif "manager" in title_lower:
            return SeniorityLevel.MANAGER
        elif "director" in title_lower:
            return SeniorityLevel.DIRECTOR
        elif "chief" in title_lower or "cto" in title_lower or "cio" in title_lower:
            return SeniorityLevel.EXECUTIVE
        elif "entry" in title_lower or "graduate" in title_lower:
            return SeniorityLevel.ENTRY
        
        # Check for experience requirements in description
        experience_patterns = [
            r'(\d+)[\+\s]*years?[\s\w]*experience',
            r'experience[\s\w]*(\d+)[\+\s]*years?'
        ]
        
        max_years = 0
        for pattern in experience_patterns:
            matches = re.finditer(pattern, description_lower, re.IGNORECASE)
            for match in matches:
                try:
                    years = int(match.group(1))
                    max_years = max(max_years, years)
                except (ValueError, IndexError):
                    pass
        
        # Map years of experience to seniority level
        if max_years > 0:
            if max_years >= 10:
                return SeniorityLevel.DIRECTOR
            elif max_years >= 7:
                return SeniorityLevel.SENIOR
            elif max_years >= 4:
                return SeniorityLevel.MID_LEVEL
            elif max_years >= 2:
                return SeniorityLevel.JUNIOR
            else:
                return SeniorityLevel.ENTRY
        
        # Check for seniority indications in combined text
        if "senior" in combined_text or "experienced" in combined_text:
            return SeniorityLevel.SENIOR
        elif "junior" in combined_text or "entry level" in combined_text:
            return SeniorityLevel.JUNIOR
        elif "mid-level" in combined_text or "intermediate" in combined_text:
            return SeniorityLevel.MID_LEVEL
        
        # Default to mid-level if not determinable
        return SeniorityLevel.MID_LEVEL
    
    @timed_function
    def calculate_ai_impact_metrics(self, job_posting: JobPosting) -> JobPosting:
        """
        Calculate additional AI impact metrics for a job posting.
        
        Args:
            job_posting: JobPosting object to calculate metrics for
            
        Returns:
            JobPosting object with updated metrics
        """
        # Initialize metrics with default values
        automation_risk = 0.0
        augmentation_potential = 0.0
        transformation_level = 0.0
        
        # Calculate automation risk based on job role and skills
        title_lower = job_posting.title.lower()
        desc_lower = job_posting.description.lower()
        
        # Roles at higher risk of automation
        high_automation_risk_keywords = [
            'data entry', 'manual testing', 'support', 'helpdesk',
            'administrative', 'clerical', 'basic reporting'
        ]
        
        for keyword in high_automation_risk_keywords:
            if keyword in title_lower or keyword in desc_lower:
                automation_risk += 0.2
        
        # Skills that increase automation risk
        automatable_skills = [
            'spreadsheet', 'data entry', 'manual testing', 'basic reporting'
        ]
        
        for skill in automatable_skills:
            if skill in desc_lower:
                automation_risk += 0.1
        
        # Calculate augmentation potential
        augmentation_potential = job_posting.ai_impact * 0.8  # Base on AI impact
        
        # Adjust based on skills that are enhanced by AI
        ai_augmentable_skills = [
            'data analysis', 'research', 'content creation', 'design',
            'customer service', 'decision making', 'coding'
        ]
        
        for skill in ai_augmentable_skills:
            if skill in desc_lower:
                augmentation_potential += 0.05
        
        # Calculate transformation level based on role innovation potential
        if job_posting.ai_impact > 0.7:
            transformation_level = 0.8
        elif job_posting.ai_impact > 0.4:
            transformation_level = 0.5
        else:
            transformation_level = 0.3
        
        # Adjust based on innovation keywords
        innovation_keywords = [
            'innovate', 'transform', 'disrupt', 'pioneer', 'revolutionary',
            'cutting-edge', 'state-of-the-art', 'breakthrough'
        ]
        
        for keyword in innovation_keywords:
            if keyword in desc_lower:
                transformation_level += 0.05
        
        # Ensure all metrics are in range 0-1
        automation_risk = min(max(automation_risk, 0.0), 1.0)
        augmentation_potential = min(max(augmentation_potential, 0.0), 1.0)
        transformation_level = min(max(transformation_level, 0.0), 1.0)
        
        # Calculate required skills count
        required_skills_count = sum(1 for skill in job_posting.skills if skill.is_required)
        
        # Calculate posting recency in days
        posting_recency = (datetime.date.today() - job_posting.date_posted).days
        
        # Create JobMetrics object
        metrics = JobMetrics(
            ai_impact_score=job_posting.ai_impact,
            ai_impact_category=AIImpactLevel(job_posting.ai_impact_category),
            salary=job_posting.salary,
            remote_percentage=job_posting.remote_percentage,
            required_skills_count=required_skills_count,
            posting_recency=posting_recency,
            automation_risk=automation_risk,
            augmentation_potential=augmentation_potential,
            transformation_level=transformation_level
        )
        
        # Update job posting with metrics
        job_posting.metrics = metrics
        
        return job_posting
    
    @timed_function
    def prepare_for_duckdb_insertion(self, job_postings: List[JobPosting]) -> pd.DataFrame:
        """
        Prepare job postings for insertion into DuckDB.
        
        Args:
            job_postings: List of JobPosting objects
            
        Returns:
            DataFrame ready for DuckDB insertion
        """
        # Convert job postings to dictionaries
        job_dicts = []
        for job in job_postings:
            job_dict = job.to_dict()
            
            # Add metrics if available
            if job.metrics:
                job_dict.update({
                    "automation_risk": job.metrics.automation_risk,
                    "augmentation_potential": job.metrics.augmentation_potential,
                    "transformation_level": job.metrics.transformation_level,
                    "required_skills_count": job.metrics.required_skills_count,
                    "posting_recency": job.metrics.posting_recency
                })
            
            job_dicts.append(job_dict)
        
        # Convert to DataFrame
        df = pd.DataFrame(job_dicts)
        
        # Ensure date columns are properly formatted
        date_columns = ['date_posted', 'application_deadline']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.date
        
        logger.info(f"Prepared {len(df)} job postings for DuckDB insertion")
        return df
    
    @timed_function
    def extract_skills_dataframe(self, job_postings: List[JobPosting]) -> pd.DataFrame:
        """
        Extract skills data from job postings for insertion into DuckDB.
        
        Args:
            job_postings: List of JobPosting objects
            
        Returns:
            DataFrame of skills data ready for DuckDB insertion
        """
        skills_data = []
        
        for job in job_postings:
            for skill in job.skills:
                skills_data.append({
                    "job_id": job.job_id,
                    "skill_name": skill.name,
                    "skill_category": skill.category.value,
                    "is_required": skill.is_required,
                    "experience_years": skill.experience_years
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(skills_data)
        logger.info(f"Extracted {len(df)} skills records for DuckDB insertion")
        return df
    
    @timed_function
    def batch_process_responses(self, responses: List[PerplexityResponse]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process multiple PerplexityAI responses in batch.
        
        Args:
            responses: List of PerplexityResponse objects
            
        Returns:
            Tuple of (jobs DataFrame, skills DataFrame) for DuckDB insertion
        """
        all_job_postings = []
        
        for response in responses:
            job_postings = self.process_perplexity_response(response)
            all_job_postings.extend(job_postings)
        
        # Prepare DataFrames for DuckDB insertion
        jobs_df = self.prepare_for_duckdb_insertion(all_job_postings)
        skills_df = self.extract_skills_dataframe(all_job_postings)
        
        return jobs_df, skills_df
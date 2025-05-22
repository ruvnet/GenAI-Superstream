"""
Data classes and enums for the advanced DuckDB implementation.

This module contains data structures used throughout the application to represent
job postings, metrics, and other entities.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import datetime


class AIImpactLevel(Enum):
    """Enumeration of AI impact levels for standardization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    TRANSFORMATIVE = "transformative"


class ContractType(Enum):
    """Enumeration of job contract types."""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"


class SeniorityLevel(Enum):
    """Enumeration of job seniority levels."""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID_LEVEL = "mid_level"
    SENIOR = "senior"
    LEAD = "lead"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class SkillCategory(Enum):
    """Enumeration of skill categories."""
    TECHNICAL = "technical"
    AI = "ai"
    SOFT = "soft"
    DOMAIN = "domain"
    TOOL = "tool"
    LANGUAGE = "language"
    FRAMEWORK = "framework"
    PLATFORM = "platform"


@dataclass
class Skill:
    """Class representing a job skill."""
    name: str
    category: SkillCategory
    is_required: bool
    experience_years: Optional[int] = None


@dataclass
class Salary:
    """Class representing salary information."""
    min_value: Optional[float]
    max_value: Optional[float]
    currency: str = "GBP"
    
    @property
    def median(self) -> Optional[float]:
        """Calculate the median salary if both min and max are available."""
        if self.min_value is not None and self.max_value is not None:
            return (self.min_value + self.max_value) / 2
        return None


@dataclass
class JobMetrics:
    """Data class for storing job metrics used in analytics."""
    ai_impact_score: float
    ai_impact_category: AIImpactLevel
    salary: Optional[Salary] = None
    remote_percentage: float = 0
    required_skills_count: int = 0
    posting_recency: int = 0  # Days since posting
    automation_risk: float = 0.0
    augmentation_potential: float = 0.0
    transformation_level: float = 0.0


@dataclass
class JobPosting:
    """Class representing a job posting."""
    job_id: str
    title: str
    company: str
    location: str
    description: str
    date_posted: datetime.date
    source: str
    ai_impact: float
    
    # Optional fields
    salary_text: Optional[str] = None
    salary: Optional[Salary] = None
    responsibilities: Optional[str] = None
    requirements: Optional[str] = None
    benefits: Optional[str] = None
    ai_impact_category: Optional[str] = None
    remote_work: bool = False
    remote_percentage: int = 0
    contract_type: Optional[ContractType] = None
    seniority_level: Optional[SeniorityLevel] = None
    application_deadline: Optional[datetime.date] = None
    source_url: Optional[str] = None
    skills: List[Skill] = None
    metrics: Optional[JobMetrics] = None
    
    def __post_init__(self):
        """Initialize derived fields after initialization."""
        # Initialize empty skills list if not provided
        if self.skills is None:
            self.skills = []
            
        # Derive AI impact category if not provided
        if self.ai_impact_category is None:
            self.set_ai_impact_category()
    
    def set_ai_impact_category(self) -> None:
        """Set the AI impact category based on the impact score."""
        if self.ai_impact < 0.25:
            self.ai_impact_category = AIImpactLevel.LOW.value
        elif self.ai_impact < 0.5:
            self.ai_impact_category = AIImpactLevel.MEDIUM.value
        elif self.ai_impact < 0.75:
            self.ai_impact_category = AIImpactLevel.HIGH.value
        else:
            self.ai_impact_category = AIImpactLevel.TRANSFORMATIVE.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the job posting to a dictionary for database insertion."""
        result = {
            "job_id": self.job_id,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "description": self.description,
            "date_posted": self.date_posted,
            "source": self.source,
            "ai_impact": self.ai_impact,
            "ai_impact_category": self.ai_impact_category,
        }
        
        # Add optional fields if they exist
        if self.salary_text:
            result["salary"] = self.salary_text
        
        if self.salary:
            result["salary_min"] = self.salary.min_value
            result["salary_max"] = self.salary.max_value
            result["salary_currency"] = self.salary.currency
        
        optional_fields = [
            "responsibilities", "requirements", "benefits", "remote_work",
            "remote_percentage", "application_deadline", "source_url"
        ]
        
        for field in optional_fields:
            value = getattr(self, field, None)
            if value is not None:
                result[field] = value
        
        if self.contract_type:
            result["contract_type"] = self.contract_type.value
            
        if self.seniority_level:
            result["seniority_level"] = self.seniority_level.value
            
        return result


@dataclass
class Company:
    """Class representing a company."""
    company_id: str
    name: str
    ai_focus_level: float = 0.5
    industry: Optional[str] = None
    company_size: Optional[str] = None
    founded_year: Optional[int] = None
    headquarters: Optional[str] = None
    website: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the company to a dictionary for database insertion."""
        result = {
            "company_id": self.company_id,
            "name": self.name,
            "ai_focus_level": self.ai_focus_level
        }
        
        optional_fields = [
            "industry", "company_size", "founded_year", "headquarters", "website"
        ]
        
        for field in optional_fields:
            value = getattr(self, field, None)
            if value is not None:
                result[field] = value
                
        return result


@dataclass
class PerplexityResponse:
    """Class representing a response from PerplexityAI."""
    query_text: str
    response_id: str
    content: str
    citations: List[str]
    data_retrieval_date: datetime.datetime = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary for database insertion."""
        return {
            "query_text": self.query_text,
            "response_id": self.response_id,
            "data_retrieval_date": self.data_retrieval_date,
            "citation_links": ", ".join(self.citations),
            "response_summary": self.content[:500]  # First 500 chars as summary
        }
"""
PerplexityAI MCP client for the advanced DuckDB implementation.

This module provides functionality to interact with the PerplexityAI MCP service
to gather AI jobs data from the UK market.
"""

import json
import logging
import datetime
import os
import uuid
from typing import Dict, List, Optional, Any, Union

from advanced.config import PERPLEXITY_CONFIG
from advanced.models.data_classes import PerplexityResponse, JobPosting, Salary, SkillCategory, Skill, ContractType, SeniorityLevel

# Set up logging
logger = logging.getLogger(__name__)


class PerplexityClient:
    """
    Client for interacting with the PerplexityAI MCP service.
    
    This class provides methods to send queries to PerplexityAI and process
    the responses to extract structured job data.
    """
    
    def __init__(self, server_name: Optional[str] = None):
        """
        Initialize the PerplexityAI client.
        
        Args:
            server_name: The name of the PerplexityAI MCP server. If None, uses the name from config or env.
        """
        # Use environment variables with fallbacks to config values
        self.server_name = server_name or os.getenv("PERPLEXITY_MCP_SERVER_NAME") or PERPLEXITY_CONFIG.get("server_name")
        self.mcp_url = os.getenv("PERPLEXITY_MCP_URL", "http://localhost:3001")
        self.system_prompt = PERPLEXITY_CONFIG.get("system_prompt")
        self.max_tokens = PERPLEXITY_CONFIG.get("max_tokens", 1000)
        self.temperature = PERPLEXITY_CONFIG.get("temperature", 0.1)
        self.return_citations = PERPLEXITY_CONFIG.get("return_citations", True)
        
        logger.info(f"PerplexityClient initialized with server: {self.server_name}, URL: {self.mcp_url}")
    
    def query_perplexity(self, query: str) -> Dict[str, Any]:
        """
        Send a query to the PerplexityAI MCP service.
        
        Args:
            query: The query text to send to PerplexityAI
            
        Returns:
            Dictionary containing the parameters for MCP tool call
            
        Note:
            This function requires the MCP tool to be used, which is handled outside
            this module by the calling code. This function sets up the query parameters.
        """
        arguments = {
            "systemContent": self.system_prompt,
            "userContent": query,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "return_citations": self.return_citations
        }
        
        logger.info(f"Prepared PerplexityAI query: {query[:100]}...")
        
        # The actual MCP call is handled outside this function
        # Return the prepared arguments
        return {
            "server_name": self.server_name,
            "tool_name": "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
            "arguments": arguments
        }
    
    def process_response(self, response_data: Dict[str, Any]) -> PerplexityResponse:
        """
        Process a response from the PerplexityAI MCP service.
        
        Args:
            response_data: The raw response data from the MCP service
            
        Returns:
            PerplexityResponse object containing the processed response
        """
        try:
            # Extract data from the response
            response_id = response_data.get("data", {}).get("response", {}).get("id", "unknown")
            content = response_data.get("data", {}).get("response", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            citations = response_data.get("data", {}).get("response", {}).get("citations", [])
            
            # Create a PerplexityResponse object
            response = PerplexityResponse(
                query_text=self.system_prompt,
                response_id=response_id,
                content=content,
                citations=citations,
                data_retrieval_date=datetime.datetime.now()
            )
            
            logger.info(f"Processed PerplexityAI response: {response_id}")
            return response
        except Exception as e:
            logger.error(f"Failed to process PerplexityAI response: {e}")
            raise
    
    def parse_jobs_data(self, response: PerplexityResponse) -> List[Dict[str, Any]]:
        """
        Parse the response content to extract structured job data.
        
        Args:
            response: The PerplexityResponse object containing job data
            
        Returns:
            List of dictionaries containing structured job data
        """
        try:
            # Try to parse the response as JSON first
            content = response.content.strip()
            if content.startswith("{") or content.startswith("["):
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("Response is not valid JSON, using fallback parsing")
            
            # Fallback: Look for tables or formatted data in the response
            jobs_data = []
            lines = content.split("\n")
            
            # Simple table detection
            table_start = None
            table_end = None
            header_row = None
            
            for i, line in enumerate(lines):
                if "|" in line and "-+-" in line.replace(" ", ""):
                    # This looks like a table separator
                    if table_start is None and i > 0:
                        table_start = i - 1
                        header_row = lines[i-1].strip().split("|")
                        header_row = [h.strip() for h in header_row if h.strip()]
                elif table_start is not None and (not "|" in line or line.strip() == ""):
                    table_end = i
                    break
            
            if table_start is not None and header_row:
                # Process table data
                for i in range(table_start + 2, table_end if table_end else len(lines)):
                    if "|" in lines[i]:
                        row = lines[i].strip().split("|")
                        row = [cell.strip() for cell in row if cell.strip()]
                        if len(row) == len(header_row):
                            job_data = dict(zip(header_row, row))
                            jobs_data.append(job_data)
            
            # If no table data was found, try to extract key-value pairs
            if not jobs_data:
                current_job = {}
                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_job:
                            jobs_data.append(current_job)
                            current_job = {}
                    elif ":" in line:
                        key, value = line.split(":", 1)
                        current_job[key.strip()] = value.strip()
                
                if current_job:
                    jobs_data.append(current_job)
            
            logger.info(f"Extracted {len(jobs_data)} job entries from response")
            return jobs_data
        
        except Exception as e:
            logger.error(f"Failed to parse jobs data: {e}")
            return []
    
    def transform_to_job_postings(self, jobs_data: List[Dict[str, Any]]) -> List[JobPosting]:
        """
        Transform parsed job data into JobPosting objects.
        
        Args:
            jobs_data: List of dictionaries containing job data
            
        Returns:
            List of JobPosting objects
        """
        job_postings = []
        
        for job_data in jobs_data:
            try:
                # Extract fields with appropriate fallbacks
                job_id = job_data.get("job_id", str(uuid.uuid4()))
                title = job_data.get("title", job_data.get("job_title", job_data.get("position", "Unknown Title")))
                company = job_data.get("company", job_data.get("company_name", job_data.get("employer", "Unknown Company")))
                location = job_data.get("location", job_data.get("city", "Unknown Location"))
                description = job_data.get("description", job_data.get("job_description", "No description available"))
                
                # Handle date posted
                date_posted_str = job_data.get("date_posted", job_data.get("posting_date", None))
                if date_posted_str:
                    try:
                        # Try various date formats
                        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y", "%d %B %Y"]:
                            try:
                                date_posted = datetime.datetime.strptime(date_posted_str, fmt).date()
                                break
                            except ValueError:
                                continue
                        else:
                            # If no format works, default to today
                            date_posted = datetime.date.today()
                    except Exception:
                        date_posted = datetime.date.today()
                else:
                    date_posted = datetime.date.today()
                
                # Extract source
                source = job_data.get("source", job_data.get("job_board", "PerplexityAI"))
                
                # Extract AI impact
                ai_impact_str = job_data.get("ai_impact", job_data.get("ai_impact_score", "0.5"))
                try:
                    ai_impact = float(ai_impact_str)
                except (ValueError, TypeError):
                    # If it's a string like "high", convert to a numeric value
                    impact_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "transformative": 0.9}
                    ai_impact = impact_map.get(ai_impact_str.lower() if isinstance(ai_impact_str, str) else "", 0.5)
                
                # Extract salary information
                salary_text = job_data.get("salary", job_data.get("salary_range", None))
                salary = None
                
                if salary_text:
                    # Try to extract min and max values
                    min_value = None
                    max_value = None
                    
                    # Look for patterns like "£30,000 - £50,000" or "$30k-$50k"
                    if "-" in salary_text:
                        parts = salary_text.split("-")
                        min_part = parts[0].strip()
                        max_part = parts[1].strip()
                        
                        # Extract numeric values
                        min_value = self._extract_salary_value(min_part)
                        max_value = self._extract_salary_value(max_part)
                    else:
                        # Single value or range specified differently
                        value = self._extract_salary_value(salary_text)
                        if value:
                            min_value = value
                            max_value = value
                    
                    if min_value or max_value:
                        currency = "GBP"  # Default for UK jobs
                        if "€" in salary_text:
                            currency = "EUR"
                        elif "$" in salary_text:
                            currency = "USD"
                        
                        salary = Salary(min_value=min_value, max_value=max_value, currency=currency)
                
                # Create the JobPosting object
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
                    # Optional fields
                    responsibilities=job_data.get("responsibilities", None),
                    requirements=job_data.get("requirements", None),
                    benefits=job_data.get("benefits", None),
                    remote_work="remote" in job_data.get("location", "").lower() or 
                               job_data.get("remote", "").lower() in ["yes", "true", "1"],
                    source_url=job_data.get("url", job_data.get("link", None))
                )
                
                # Extract and add skills if available
                skills_text = job_data.get("skills", job_data.get("required_skills", ""))
                if skills_text and isinstance(skills_text, str):
                    skill_list = [s.strip() for s in skills_text.split(",")]
                    for skill_name in skill_list:
                        if skill_name:
                            # Determine if it's likely an AI skill
                            category = SkillCategory.AI if any(ai_term in skill_name.lower() 
                                                            for ai_term in ["ai", "ml", "machine learning", "deep learning", 
                                                                           "nlp", "neural", "language model"]) \
                                    else SkillCategory.TECHNICAL
                            
                            job_posting.skills.append(Skill(
                                name=skill_name,
                                category=category,
                                is_required=True
                            ))
                
                job_postings.append(job_posting)
                
            except Exception as e:
                logger.error(f"Failed to transform job data: {e}")
                continue
        
        logger.info(f"Transformed {len(job_postings)} job postings")
        return job_postings
    
    def _extract_salary_value(self, salary_text: str) -> Optional[float]:
        """
        Extract a numeric salary value from text.
        
        Args:
            salary_text: String containing salary information
            
        Returns:
            Float value of the salary, or None if not extractable
        """
        if not salary_text:
            return None
        
        # Remove currency symbols and commas
        for symbol in ["£", "$", "€", ",", "GBP", "USD", "EUR"]:
            salary_text = salary_text.replace(symbol, "")
        
        # Look for numeric part
        import re
        numeric_matches = re.findall(r'\d+\.?\d*', salary_text)
        if not numeric_matches:
            return None
        
        value = float(numeric_matches[0])
        
        # Check for 'k' or 'K' to indicate thousands
        if "k" in salary_text.lower():
            value *= 1000
            
        return value


# Function to create a query for UK AI jobs data
def create_uk_ai_jobs_query(specific_role: Optional[str] = None, 
                           location: Optional[str] = None,
                           timeframe: Optional[str] = None) -> str:
    """
    Create a well-structured query for PerplexityAI about UK AI jobs.
    
    Args:
        specific_role: Optional specific AI role to focus on (e.g., "Machine Learning Engineer")
        location: Optional specific location in the UK (e.g., "London")
        timeframe: Optional timeframe for the data (e.g., "2025", "last 3 months")
        
    Returns:
        Formatted query string
    """
    query = "What are the latest trends in AI technical jobs in the UK?"
    
    if specific_role:
        query += f" Focus specifically on {specific_role} roles."
    
    if location:
        query += f" Concentrate on jobs in {location}."
    
    if timeframe:
        query += f" Provide data from {timeframe}."
    
    query += (
        " Please provide comprehensive data on job titles, companies, locations, "
        "salary ranges, job descriptions, AI impact metrics, posting dates, and data sources. "
        "Structure your response as a table or easily parsable format for database insertion."
    )
    
    return query


def create_jobs_database_query(job_data: Union[List[JobPosting], JobPosting]) -> Dict[str, Any]:
    """
    Transform job data into a format suitable for database insertion.
    
    Args:
        job_data: A JobPosting object or list of JobPosting objects
        
    Returns:
        Dictionary containing database-ready job data
    """
    if isinstance(job_data, list):
        return [job.to_dict() for job in job_data]
    else:
        return job_data.to_dict()
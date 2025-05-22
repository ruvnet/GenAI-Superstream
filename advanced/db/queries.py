"""
Database query operations for the advanced DuckDB implementation.

This module provides classes and functions for interacting with the database,
including querying, inserting, updating, and analyzing job data.
"""

import datetime
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd

from advanced.models.schemas import ALL_TABLES, INDEX_DEFINITIONS, COMMON_QUERIES
from advanced.models.data_classes import JobPosting, Skill, Company, PerplexityResponse
from advanced.db.connection import connection_manager, timed
from advanced.config import DEFAULT_QUERY_PARAMS

# Set up logging
logger = logging.getLogger(__name__)


class JobsDatabase:
    """
    Advanced class for managing the UK jobs database with enhanced features.
    
    This class provides methods for database schema management, data insertion,
    querying, and analytics functionality.
    """
    
    def __init__(self):
        """Initialize the JobsDatabase with a connection to DuckDB."""
        self.connection = connection_manager
    
    @timed
    def initialize_schema(self) -> None:
        """
        Initialize the database schema with all required tables and indexes.
        
        Creates all tables and indexes defined in the schemas module.
        """
        try:
            conn = self.connection.connect()
            
            # Create all tables
            for table_sql in ALL_TABLES:
                conn.execute(table_sql)
            
            # Create all indexes
            for index_sql in INDEX_DEFINITIONS:
                try:
                    conn.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")
            
            logger.info("Advanced database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise
    
    @timed
    def insert_job(self, job: JobPosting) -> None:
        """
        Insert a job posting into the database.
        
        Args:
            job: JobPosting object containing job data
        """
        try:
            conn = self.connection.connect()
            
            # Start transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Convert job to dictionary for insertion
            job_dict = job.to_dict()
            
            # Prepare SQL for job_postings
            fields = ', '.join(job_dict.keys())
            placeholders = ', '.join(['?' for _ in job_dict])
            
            sql = f"INSERT INTO job_postings ({fields}) VALUES ({placeholders})"
            
            # Execute the insert for job_postings
            conn.execute(sql, tuple(job_dict.values()))
            
            # Insert skills if any
            for skill in job.skills:
                conn.execute("""
                    INSERT INTO job_skills 
                    (job_id, skill_name, skill_category, is_required, experience_years)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    job.job_id, 
                    skill.name, 
                    skill.category.value, 
                    skill.is_required,
                    skill.experience_years
                ))
            
            # Insert AI impact metrics if available
            if job.metrics:
                conn.execute("""
                    INSERT INTO ai_impact_metrics
                    (job_id, automation_risk, augmentation_potential, transformation_level, analysis_date, analysis_method)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    job.job_id,
                    job.metrics.automation_risk,
                    job.metrics.augmentation_potential,
                    job.metrics.transformation_level,
                    datetime.date.today(),
                    'data_class_derived'
                ))
            
            # Record the job creation in history
            conn.execute("""
                INSERT INTO job_history
                (job_id, event_type, field_name, new_value)
                VALUES (?, ?, ?, ?)
            """, (
                job.job_id,
                'created',
                'all',
                'Initial job posting creation'
            ))
            
            # Commit transaction
            conn.execute("COMMIT")
            
            logger.info(f"Inserted job posting with ID: {job.job_id}")
        except Exception as e:
            # Rollback on error
            conn.execute("ROLLBACK")
            logger.error(f"Failed to insert job posting: {e}")
            raise
    
    @timed
    def insert_jobs_batch(self, jobs: List[JobPosting]) -> Tuple[int, int]:
        """
        Insert multiple job postings in batch with enhanced error handling.
        
        Args:
            jobs: List of JobPosting objects
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0
        
        conn = self.connection.connect()
        try:
            # Start a transaction for better performance and atomicity
            conn.execute("BEGIN TRANSACTION")
            
            for job in jobs:
                try:
                    self.insert_job(job)
                    successful += 1
                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to insert job {job.job_id}: {e}")
            
            # Commit the transaction
            conn.execute("COMMIT")
            
            logger.info(f"Batch insert complete: {successful} succeeded, {failed} failed")
            return (successful, failed)
        except Exception as e:
            # Rollback on error
            conn.execute("ROLLBACK")
            logger.error(f"Failed to insert batch of job postings, transaction rolled back: {e}")
            raise
    
    @timed
    def insert_perplexity_source(self, response: PerplexityResponse) -> int:
        """
        Insert a PerplexityAI response as a data source.
        
        Args:
            response: PerplexityResponse object
            
        Returns:
            Inserted source ID
        """
        try:
            conn = self.connection.connect()
            
            response_dict = response.to_dict()
            fields = ', '.join(response_dict.keys())
            placeholders = ', '.join(['?' for _ in response_dict])
            
            sql = f"INSERT INTO perplexity_data_sources ({fields}) VALUES ({placeholders}) RETURNING source_id"
            
            result = conn.execute(sql, tuple(response_dict.values())).fetchone()
            source_id = result[0] if result else None
            
            logger.info(f"Inserted PerplexityAI response as data source ID: {source_id}")
            return source_id
        except Exception as e:
            logger.error(f"Failed to insert PerplexityAI response: {e}")
            raise
    
    @timed
    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a job posting by its ID with enhanced data enrichment.
        
        Args:
            job_id: The unique identifier of the job posting
            
        Returns:
            Dictionary containing job posting data with additional related information
        """
        try:
            conn = self.connection.connect()
            
            # Get the base job posting
            job_query = COMMON_QUERIES["get_job_by_id"]
            result = conn.execute(job_query, (job_id,)).fetchall()
            
            if not result:
                return None
            
            # Convert to dictionary
            columns = [desc[0] for desc in conn.description]
            job = dict(zip(columns, result[0]))
            
            # Enhance with skills information
            skills_query = COMMON_QUERIES["get_job_skills"]
            skills = conn.execute(skills_query, (job_id,)).fetchall()
            job['skills'] = [
                {'name': s[0], 'category': s[1], 'required': s[2]} 
                for s in skills
            ]
            
            # Enhance with AI impact metrics
            metrics_query = COMMON_QUERIES["get_job_ai_metrics"]
            metrics = conn.execute(metrics_query, (job_id,)).fetchone()
            if metrics:
                job['ai_metrics'] = {
                    'automation_risk': metrics[0],
                    'augmentation_potential': metrics[1],
                    'transformation_level': metrics[2]
                }
            
            # Enhance with company information
            company_query = COMMON_QUERIES["get_company_by_job_id"]
            company = conn.execute(company_query, (job_id,)).fetchone()
            if company:
                company_columns = [desc[0] for desc in conn.description]
                job['company_details'] = dict(zip(company_columns, company))
            
            return job
        except Exception as e:
            logger.error(f"Failed to get job by ID {job_id}: {e}")
            raise
    
    @timed
    def search_jobs(self, 
                   title: Optional[str] = None,
                   company: Optional[str] = None,
                   location: Optional[str] = None,
                   min_ai_impact: Optional[float] = None,
                   max_ai_impact: Optional[float] = None,
                   ai_impact_category: Optional[str] = None,
                   required_skills: Optional[List[str]] = None,
                   remote_work: Optional[bool] = None,
                   min_salary: Optional[float] = None,
                   max_salary: Optional[float] = None,
                   start_date: Optional[Union[str, datetime.date]] = None,
                   end_date: Optional[Union[str, datetime.date]] = None,
                   seniority_level: Optional[str] = None,
                   contract_type: Optional[str] = None,
                   limit: int = DEFAULT_QUERY_PARAMS["limit"],
                   offset: int = DEFAULT_QUERY_PARAMS["offset"]) -> List[Dict[str, Any]]:
        """
        Search for job postings with advanced filtering options.
        
        Args:
            title: Filter by job title (partial match)
            company: Filter by company name (partial match)
            location: Filter by location (partial match)
            min_ai_impact: Minimum AI impact score
            max_ai_impact: Maximum AI impact score
            ai_impact_category: AI impact category (low, medium, high, transformative)
            required_skills: List of required skills
            remote_work: Filter by remote work availability
            min_salary: Minimum salary
            max_salary: Maximum salary
            start_date: Start date for date_posted range
            end_date: End date for date_posted range
            seniority_level: Filter by seniority level
            contract_type: Filter by contract type
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            
        Returns:
            List of dictionaries containing matching job postings
        """
        conditions = []
        parameters = []
        
        # Helper function to add filter condition
        def add_filter(condition, parameter):
            conditions.append(condition)
            parameters.append(parameter)
        
        # Apply all filters
        if title:
            add_filter("title ILIKE ?", f"%{title}%")
        
        if company:
            add_filter("company ILIKE ?", f"%{company}%")
        
        if location:
            add_filter("location ILIKE ?", f"%{location}%")
        
        if min_ai_impact is not None:
            add_filter("ai_impact >= ?", min_ai_impact)
        
        if max_ai_impact is not None:
            add_filter("ai_impact <= ?", max_ai_impact)
            
        if ai_impact_category:
            add_filter("ai_impact_category = ?", ai_impact_category)
            
        if remote_work is not None:
            add_filter("remote_work = ?", remote_work)
            
        if min_salary is not None:
            add_filter("salary_min >= ?", min_salary)
            
        if max_salary is not None:
            add_filter("salary_max <= ?", max_salary)
            
        if seniority_level:
            add_filter("seniority_level = ?", seniority_level)
            
        if contract_type:
            add_filter("contract_type = ?", contract_type)
        
        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
            add_filter("date_posted >= ?", start_date)
        
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
            add_filter("date_posted <= ?", end_date)
        
        # Handle skills filtering with a subquery
        if required_skills and len(required_skills) > 0:
            skills_placeholders = ', '.join(['?' for _ in required_skills])
            skill_condition = f"""
                job_id IN (
                    SELECT job_id 
                    FROM job_skills 
                    WHERE skill_name IN ({skills_placeholders})
                    GROUP BY job_id 
                    HAVING COUNT(DISTINCT skill_name) = ?
                )
            """
            conditions.append(skill_condition)
            parameters.extend(required_skills)
            parameters.append(len(required_skills))  # Must match all skills
        
        # Build the final query
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM job_postings 
            WHERE {where_clause} 
            ORDER BY date_posted DESC 
            LIMIT ? OFFSET ?
        """
        
        # Add pagination parameters
        parameters.append(limit)
        parameters.append(offset)
        
        try:
            conn = self.connection.connect()
            result = conn.execute(query, tuple(parameters)).fetchall()
            
            # Convert to dictionaries
            columns = [desc[0] for desc in conn.description]
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"Failed to search jobs: {e}")
            raise
    
    @timed
    def to_dataframe(self, query: str = "SELECT * FROM job_postings", 
                    parameters: Optional[tuple] = None) -> pd.DataFrame:
        """
        Export query results to a pandas DataFrame.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            
        Returns:
            pandas DataFrame containing query results
        """
        try:
            conn = self.connection.connect()
            return conn.execute(query, parameters).fetch_df() if parameters else conn.execute(query).fetch_df()
        except Exception as e:
            logger.error(f"Failed to export to DataFrame: {e}")
            raise
    
    @timed
    def record_data_quality_metrics(self, table_name: str, column_name: Optional[str] = None) -> None:
        """
        Record data quality metrics for a table or column.
        
        Args:
            table_name: Name of the table to analyze
            column_name: Optional column name to analyze (if None, analyzes all columns)
        """
        try:
            conn = self.connection.connect()
            
            if column_name:
                # Calculate metrics for a specific column
                query = f"""
                    INSERT INTO data_quality_metrics
                    (table_name, column_name, completeness, check_date)
                    SELECT
                        '{table_name}' as table_name,
                        '{column_name}' as column_name,
                        (COUNT({column_name}) * 100.0 / COUNT(*)) as completeness,
                        CURRENT_TIMESTAMP as check_date
                    FROM {table_name}
                """
                conn.execute(query)
            else:
                # Get all columns in the table
                columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                column_names = [col[1] for col in columns if col[1] != 'job_id']
                
                # Calculate metrics for each column
                for col in column_names:
                    query = f"""
                        INSERT INTO data_quality_metrics
                        (table_name, column_name, completeness, check_date)
                        SELECT
                            '{table_name}' as table_name,
                            '{col}' as column_name,
                            (COUNT({col}) * 100.0 / COUNT(*)) as completeness,
                            CURRENT_TIMESTAMP as check_date
                        FROM {table_name}
                    """
                    conn.execute(query)
            
            logger.info(f"Recorded data quality metrics for {table_name}")
        except Exception as e:
            logger.error(f"Failed to record data quality metrics: {e}")
            raise
    
    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()
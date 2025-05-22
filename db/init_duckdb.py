#!/usr/bin/env python3
"""
DuckDB initialization and utility script for UK jobs data analytics.
This module provides functionality for creating, managing, and querying a DuckDB database
that stores job postings with a focus on AI's impact on technical roles in the UK.
"""

import os
import datetime
import logging
from typing import Dict, List, Optional, Union, Any

import duckdb
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JobsDatabase:
    """
    A class to manage the UK jobs database using DuckDB.
    
    This class provides methods to initialize the database schema,
    insert job data, query job records, and export data to pandas
    DataFrames for integration with scikit-learn.
    """
    
    def __init__(self, db_path: str = "uk_jobs.duckdb"):
        """
        Initialize the JobsDatabase with a connection to DuckDB.
        
        Args:
            db_path: Path to the DuckDB database file. Defaults to "uk_jobs.duckdb".
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
        
    def _connect(self) -> None:
        """Establish a connection to the DuckDB database."""
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"Connected to database at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def initialize_schema(self) -> None:
        """
        Initialize the database schema for job postings.
        
        Creates the job_postings table if it doesn't exist.
        """
        try:
            # Create the job_postings table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS job_postings (
                    job_id VARCHAR PRIMARY KEY,
                    title VARCHAR NOT NULL,
                    company VARCHAR NOT NULL,
                    location VARCHAR NOT NULL,
                    salary VARCHAR,  -- Using VARCHAR as salary might be a range or contain currency symbols
                    description TEXT,
                    ai_impact FLOAT,  -- A score/metric indicating AI's impact on this role (e.g., 0-1)
                    date_posted DATE NOT NULL,
                    source VARCHAR NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index on common search fields
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON job_postings(title)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_company ON job_postings(company)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_location ON job_postings(location)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_date_posted ON job_postings(date_posted)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_impact ON job_postings(ai_impact)")
            
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise
    
    def insert_job(self, job_data: Dict[str, Any]) -> None:
        """
        Insert a single job posting into the database.
        
        Args:
            job_data: Dictionary containing job posting data with keys matching table columns.
                      Required keys: job_id, title, company, location, date_posted, source
        """
        required_fields = ['job_id', 'title', 'company', 'location', 'date_posted', 'source']
        for field in required_fields:
            if field not in job_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Convert date_posted to DATE if it's a string
        if isinstance(job_data.get('date_posted'), str):
            job_data['date_posted'] = datetime.datetime.strptime(
                job_data['date_posted'], '%Y-%m-%d').date()
        
        try:
            # Prepare the SQL statement
            fields = ', '.join(job_data.keys())
            placeholders = ', '.join(['?' for _ in job_data])
            
            sql = f"INSERT INTO job_postings ({fields}) VALUES ({placeholders})"
            
            # Execute the insert
            self.conn.execute(sql, list(job_data.values()))
            logger.info(f"Inserted job posting with ID: {job_data['job_id']}")
        except Exception as e:
            logger.error(f"Failed to insert job posting: {e}")
            raise
    
    def insert_jobs_batch(self, jobs_data: List[Dict[str, Any]]) -> None:
        """
        Insert multiple job postings in batch.
        
        Args:
            jobs_data: List of dictionaries containing job posting data.
        """
        try:
            for job_data in jobs_data:
                self.insert_job(job_data)
            logger.info(f"Inserted {len(jobs_data)} job postings in batch")
        except Exception as e:
            logger.error(f"Failed to insert batch of job postings: {e}")
            raise
    
    def insert_jobs_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Insert job postings from a pandas DataFrame.
        
        Args:
            df: DataFrame containing job posting data with columns matching table columns.
        """
        try:
            # Convert DataFrame to a list of dictionaries
            jobs_data = df.to_dict(orient='records')
            self.insert_jobs_batch(jobs_data)
            logger.info(f"Inserted {len(jobs_data)} job postings from DataFrame")
        except Exception as e:
            logger.error(f"Failed to insert job postings from DataFrame: {e}")
            raise
    
    def query_jobs(self, query: str, parameters: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom query against the job_postings table.
        
        Args:
            query: SQL query string.
            parameters: List of parameters for the SQL query.
            
        Returns:
            List of dictionaries containing query results.
        """
        try:
            if parameters:
                result = self.conn.execute(query, parameters).fetchall()
            else:
                result = self.conn.execute(query).fetchall()
            
            # Convert result to list of dictionaries
            columns = [desc[0] for desc in self.conn.description]
            return [dict(zip(columns, row)) for row in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_job_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a job posting by its ID.
        
        Args:
            job_id: The unique identifier of the job posting.
            
        Returns:
            Dictionary containing job posting data or None if not found.
        """
        try:
            result = self.query_jobs(
                "SELECT * FROM job_postings WHERE job_id = ?", 
                [job_id]
            )
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to get job by ID {job_id}: {e}")
            raise
    
    def search_jobs(self, 
                   title: Optional[str] = None,
                   company: Optional[str] = None,
                   location: Optional[str] = None,
                   min_ai_impact: Optional[float] = None,
                   max_ai_impact: Optional[float] = None,
                   start_date: Optional[Union[str, datetime.date]] = None,
                   end_date: Optional[Union[str, datetime.date]] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for job postings with various filters.
        
        Args:
            title: Filter by job title (partial match).
            company: Filter by company name (partial match).
            location: Filter by location (partial match).
            min_ai_impact: Minimum AI impact score.
            max_ai_impact: Maximum AI impact score.
            start_date: Start date for date_posted range.
            end_date: End date for date_posted range.
            limit: Maximum number of results to return.
            
        Returns:
            List of dictionaries containing matching job postings.
        """
        conditions = []
        parameters = []
        
        if title:
            conditions.append("title ILIKE ?")
            parameters.append(f"%{title}%")
        
        if company:
            conditions.append("company ILIKE ?")
            parameters.append(f"%{company}%")
        
        if location:
            conditions.append("location ILIKE ?")
            parameters.append(f"%{location}%")
        
        if min_ai_impact is not None:
            conditions.append("ai_impact >= ?")
            parameters.append(min_ai_impact)
        
        if max_ai_impact is not None:
            conditions.append("ai_impact <= ?")
            parameters.append(max_ai_impact)
        
        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
            conditions.append("date_posted >= ?")
            parameters.append(start_date)
        
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
            conditions.append("date_posted <= ?")
            parameters.append(end_date)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM job_postings WHERE {where_clause} ORDER BY date_posted DESC LIMIT {limit}"
        
        return self.query_jobs(query, parameters)
    
    def to_dataframe(self, query: str = "SELECT * FROM job_postings", parameters: Optional[List[Any]] = None) -> pd.DataFrame:
        """
        Export query results to a pandas DataFrame for scikit-learn integration.
        
        Args:
            query: SQL query string. Defaults to selecting all job postings.
            parameters: List of parameters for the SQL query.
            
        Returns:
            pandas DataFrame containing query results.
        """
        try:
            if parameters:
                return self.conn.execute(query, parameters).fetch_df()
            return self.conn.execute(query).fetch_df()
        except Exception as e:
            logger.error(f"Failed to export to DataFrame: {e}")
            raise
    
    def get_ai_impact_distribution(self) -> pd.DataFrame:
        """
        Get the distribution of AI impact scores for analysis.
        
        Returns:
            DataFrame with AI impact distribution statistics.
        """
        query = """
        SELECT 
            COUNT(*) as total_jobs,
            AVG(ai_impact) as avg_impact,
            MIN(ai_impact) as min_impact,
            MAX(ai_impact) as max_impact,
            STDDEV(ai_impact) as std_impact,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ai_impact) as q1_impact,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ai_impact) as median_impact,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ai_impact) as q3_impact
        FROM job_postings
        WHERE ai_impact IS NOT NULL
        """
        return self.to_dataframe(query)
    
    def get_top_companies_by_ai_jobs(self, limit: int = 10) -> pd.DataFrame:
        """
        Get the top companies with the highest number of AI-impacted jobs.
        
        Args:
            limit: Maximum number of companies to return.
            
        Returns:
            DataFrame with companies and their job counts.
        """
        query = f"""
        SELECT 
            company,
            COUNT(*) as job_count,
            AVG(ai_impact) as avg_ai_impact
        FROM job_postings
        WHERE ai_impact > 0.5
        GROUP BY company
        ORDER BY job_count DESC, avg_ai_impact DESC
        LIMIT {limit}
        """
        return self.to_dataframe(query)
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def main():
    """
    Main function to demonstrate database initialization and usage.
    """
    # Example usage
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uk_jobs.duckdb")
    db = JobsDatabase(db_path)
    
    # Initialize schema
    db.initialize_schema()
    
    # Example of inserting a job posting
    sample_job = {
        "job_id": "JOB-2023-001",
        "title": "Machine Learning Engineer",
        "company": "UK Tech Ltd",
        "location": "London, UK",
        "salary": "£60,000 - £80,000",
        "description": "We are looking for a Machine Learning Engineer to join our team...",
        "ai_impact": 0.85,
        "date_posted": "2023-05-15",
        "source": "company_website"
    }
    
    try:
        db.insert_job(sample_job)
        
        # Example of querying the database
        result = db.search_jobs(title="Machine Learning")
        print(f"Found {len(result)} jobs matching 'Machine Learning'")
        
        # Example of exporting to DataFrame
        df = db.to_dataframe()
        print(f"Exported {len(df)} jobs to DataFrame")
        print(df.head())
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
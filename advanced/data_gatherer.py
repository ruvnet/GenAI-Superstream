#!/usr/bin/env python3
"""
Data gatherer script for the PerplexityAI MCP integration.

This script provides a command-line interface for gathering AI job data from
PerplexityAI and inserting it into the DuckDB database.
"""

import argparse
import json
import sys
import os
import datetime
from typing import List, Dict, Optional, Any

# Add parent directory to path to handle imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced.config import DB_CONFIG, PERPLEXITY_CONFIG
from advanced.perplexity.client import PerplexityClient, create_uk_ai_jobs_query
from advanced.perplexity.data_processor import DataProcessor
from advanced.db.connection import connection_manager
from advanced.models.schemas import ALL_TABLES, INDEX_DEFINITIONS
from advanced.models.data_classes import JobPosting
from advanced.utils.logging import setup_logging, LoggedOperation


# Set up logging
logger = setup_logging(__name__)


class DataGatherer:
    """
    Main class for gathering data from PerplexityAI and storing it in DuckDB.
    """
    
    def __init__(self, dry_run: bool = False):
        """
        Initialize the data gatherer.
        
        Args:
            dry_run: If True, show what would be done without executing
        """
        self.dry_run = dry_run
        self.client = PerplexityClient()
        self.processor = DataProcessor()
        self.db = connection_manager
        
        if self.dry_run:
            logger.info("Running in dry-run mode - no changes will be made")
    
    def initialize_database(self) -> None:
        """Initialize the database with required tables and indexes."""
        with LoggedOperation("Database initialization", logger):
            if self.dry_run:
                logger.info("Would create the following tables:")
                for table_sql in ALL_TABLES:
                    table_name = table_sql.split("CREATE TABLE IF NOT EXISTS")[1].split("(")[0].strip()
                    logger.info(f"  - {table_name}")
                logger.info("Would create indexes for optimized querying")
                return
            
            with self.db:
                # Create tables
                for table_sql in ALL_TABLES:
                    self.db.execute(table_sql)
                    logger.info(f"Created/verified table from SQL: {table_sql[:50]}...")
                
                # Create indexes
                for index_sql in INDEX_DEFINITIONS:
                    self.db.execute(index_sql)
                    logger.info(f"Created/verified index: {index_sql}")
                
                logger.info("Database initialization completed successfully")
    
    def gather_data(self, 
                   query: Optional[str] = None,
                   role: Optional[str] = None,
                   location: Optional[str] = None,
                   timeframe: Optional[str] = None,
                   batch_size: int = 10) -> List[JobPosting]:
        """
        Gather data from PerplexityAI.
        
        Args:
            query: Custom query string for PerplexityAI
            role: Specific AI role to focus on
            location: Specific UK location
            timeframe: Timeframe for the data
            batch_size: Number of jobs to process in each batch
            
        Returns:
            List of JobPosting objects
        """
        with LoggedOperation("Data gathering from PerplexityAI", logger):
            # Create query
            if query:
                perplexity_query = query
                logger.info(f"Using custom query: {query}")
            else:
                perplexity_query = create_uk_ai_jobs_query(
                    specific_role=role,
                    location=location,
                    timeframe=timeframe
                )
                logger.info(f"Generated query: {perplexity_query}")
            
            if self.dry_run:
                logger.info("Would send the following query to PerplexityAI:")
                logger.info(f"  Query: {perplexity_query}")
                logger.info(f"  Batch size: {batch_size}")
                return []
            
            # Prepare the query parameters
            query_params = self.client.query_perplexity(perplexity_query)
            
            # Note: The actual MCP tool call would be handled by the calling code
            # For now, we'll return the query parameters
            logger.info("Query parameters prepared for PerplexityAI MCP tool")
            logger.info(f"Server: {query_params['server_name']}")
            logger.info(f"Tool: {query_params['tool_name']}")
            
            # Return empty list as actual data would come from MCP tool response
            return []
    
    def process_perplexity_response(self, response_data: Dict[str, Any]) -> List[JobPosting]:
        """
        Process a response from PerplexityAI MCP tool.
        
        Args:
            response_data: The raw response data from the MCP tool
            
        Returns:
            List of processed JobPosting objects
        """
        with LoggedOperation("Processing PerplexityAI response", logger):
            # Process the response
            perplexity_response = self.client.process_response(response_data)
            
            # Parse jobs data
            jobs_data = self.client.parse_jobs_data(perplexity_response)
            logger.info(f"Parsed {len(jobs_data)} job entries from response")
            
            # Transform to JobPosting objects
            job_postings = self.client.transform_to_job_postings(jobs_data)
            logger.info(f"Transformed {len(job_postings)} job postings")
            
            # Process with data processor for additional enrichment
            processed_postings = []
            for posting in job_postings:
                processed = self.processor.process_job_posting(posting)
                processed_postings.append(processed)
            
            return processed_postings
    
    def insert_jobs_to_database(self, job_postings: List[JobPosting]) -> int:
        """
        Insert job postings into the database.
        
        Args:
            job_postings: List of JobPosting objects to insert
            
        Returns:
            Number of jobs successfully inserted
        """
        with LoggedOperation(f"Inserting {len(job_postings)} jobs to database", logger):
            if self.dry_run:
                logger.info(f"Would insert {len(job_postings)} job postings:")
                for i, job in enumerate(job_postings[:3]):  # Show first 3 as examples
                    logger.info(f"  {i+1}. {job.title} at {job.company} ({job.location})")
                if len(job_postings) > 3:
                    logger.info(f"  ... and {len(job_postings) - 3} more")
                return len(job_postings)
            
            inserted_count = 0
            
            with self.db:
                for job in job_postings:
                    try:
                        # Insert job posting
                        self._insert_job_posting(job)
                        
                        # Insert skills
                        self._insert_job_skills(job)
                        
                        # Insert AI impact metrics if available
                        if hasattr(job, 'ai_metrics') and job.ai_metrics:
                            self._insert_ai_metrics(job)
                        
                        inserted_count += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to insert job {job.job_id}: {e}")
                        continue
                
                logger.info(f"Successfully inserted {inserted_count} out of {len(job_postings)} jobs")
            
            return inserted_count
    
    def _insert_job_posting(self, job: JobPosting) -> None:
        """Insert a single job posting into the database."""
        insert_sql = """
        INSERT OR REPLACE INTO job_postings (
            job_id, title, company, location, salary, salary_min, salary_max,
            salary_currency, description, responsibilities, requirements, benefits,
            ai_impact, ai_impact_category, remote_work, remote_percentage,
            contract_type, seniority_level, date_posted, application_deadline,
            source, source_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Prepare values
        values = (
            job.job_id,
            job.title,
            job.company,
            job.location,
            job.salary_text,
            job.salary.min_value if job.salary else None,
            job.salary.max_value if job.salary else None,
            job.salary.currency if job.salary else None,
            job.description,
            job.responsibilities,
            job.requirements,
            job.benefits,
            job.ai_impact,
            job.ai_impact_category.value if hasattr(job, 'ai_impact_category') and job.ai_impact_category else None,
            job.remote_work,
            getattr(job, 'remote_percentage', None),
            job.contract_type.value if hasattr(job, 'contract_type') and job.contract_type else None,
            job.seniority_level.value if hasattr(job, 'seniority_level') and job.seniority_level else None,
            job.date_posted,
            getattr(job, 'application_deadline', None),
            job.source,
            job.source_url
        )
        
        self.db.execute(insert_sql, values)
    
    def _insert_job_skills(self, job: JobPosting) -> None:
        """Insert skills for a job posting."""
        if not job.skills:
            return
        
        insert_sql = """
        INSERT OR REPLACE INTO job_skills (
            job_id, skill_name, skill_category, is_required, experience_years
        ) VALUES (?, ?, ?, ?, ?)
        """
        
        for skill in job.skills:
            values = (
                job.job_id,
                skill.name,
                skill.category.value if skill.category else None,
                skill.is_required,
                getattr(skill, 'experience_years', None)
            )
            self.db.execute(insert_sql, values)
    
    def _insert_ai_metrics(self, job: JobPosting) -> None:
        """Insert AI impact metrics for a job posting."""
        insert_sql = """
        INSERT INTO ai_impact_metrics (
            job_id, automation_risk, augmentation_potential, transformation_level,
            analysis_date, analysis_method
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        
        values = (
            job.job_id,
            getattr(job.ai_metrics, 'automation_risk', None),
            getattr(job.ai_metrics, 'augmentation_potential', None),
            getattr(job.ai_metrics, 'transformation_level', None),
            datetime.date.today(),
            'PerplexityAI Analysis'
        )
        
        self.db.execute(insert_sql, values)
    
    def show_statistics(self) -> None:
        """Show statistics about the current database content."""
        with LoggedOperation("Fetching database statistics", logger):
            with self.db:
                # Total jobs
                total_jobs = self.db.fetch_all("SELECT COUNT(*) FROM job_postings")[0][0]
                logger.info(f"Total job postings: {total_jobs}")
                
                # Jobs by AI impact category
                category_stats = self.db.fetch_all("""
                    SELECT ai_impact_category, COUNT(*) as count
                    FROM job_postings
                    WHERE ai_impact_category IS NOT NULL
                    GROUP BY ai_impact_category
                    ORDER BY count DESC
                """)
                
                if category_stats:
                    logger.info("Jobs by AI impact category:")
                    for category, count in category_stats:
                        logger.info(f"  - {category}: {count}")
                
                # Top companies
                company_stats = self.db.fetch_all("""
                    SELECT company, COUNT(*) as count
                    FROM job_postings
                    GROUP BY company
                    ORDER BY count DESC
                    LIMIT 5
                """)
                
                if company_stats:
                    logger.info("Top 5 companies by job postings:")
                    for company, count in company_stats:
                        logger.info(f"  - {company}: {count}")
                
                # Top skills
                skill_stats = self.db.fetch_all("""
                    SELECT skill_name, COUNT(*) as count
                    FROM job_skills
                    GROUP BY skill_name
                    ORDER BY count DESC
                    LIMIT 10
                """)
                
                if skill_stats:
                    logger.info("Top 10 skills:")
                    for skill, count in skill_stats:
                        logger.info(f"  - {skill}: {count}")


def main():
    """Main entry point for the data gatherer script."""
    parser = argparse.ArgumentParser(
        description="Gather AI job data from PerplexityAI and store in DuckDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize the database
  python data_gatherer.py --init
  
  # Gather data with default query
  python data_gatherer.py --gather
  
  # Gather data for specific role and location
  python data_gatherer.py --gather --role "Machine Learning Engineer" --location "London"
  
  # Use custom query
  python data_gatherer.py --gather --query "What are the highest paying AI jobs in Manchester?"
  
  # Dry run to see what would be done
  python data_gatherer.py --gather --dry-run
        """
    )
    
    # Actions
    parser.add_argument('--init', action='store_true',
                       help='Initialize the database with required tables and indexes')
    parser.add_argument('--gather', action='store_true',
                       help='Gather data from PerplexityAI')
    parser.add_argument('--stats', action='store_true',
                       help='Show statistics about the current database content')
    
    # Query options
    parser.add_argument('--query', type=str,
                       help='Custom query string for PerplexityAI')
    parser.add_argument('--role', type=str,
                       help='Specific AI role to focus on (e.g., "Machine Learning Engineer")')
    parser.add_argument('--location', type=str,
                       help='Specific UK location (e.g., "London")')
    parser.add_argument('--timeframe', type=str,
                       help='Timeframe for the data (e.g., "last 3 months")')
    
    # Processing options
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of jobs to process in each batch (default: 10)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually executing')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate that at least one action is specified
    if not any([args.init, args.gather, args.stats]):
        parser.error("At least one action (--init, --gather, or --stats) must be specified")
    
    # Create data gatherer
    gatherer = DataGatherer(dry_run=args.dry_run)
    
    try:
        # Execute requested actions
        if args.init:
            gatherer.initialize_database()
        
        if args.gather:
            # Note: In a real implementation, this would need to integrate with
            # the MCP tool system to actually fetch data from PerplexityAI
            logger.info("Gathering data from PerplexityAI...")
            
            # Prepare the query
            job_postings = gatherer.gather_data(
                query=args.query,
                role=args.role,
                location=args.location,
                timeframe=args.timeframe,
                batch_size=args.batch_size
            )
            
            if not args.dry_run:
                logger.warning(
                    "Note: Actual PerplexityAI integration requires MCP tool execution. "
                    "Please use the prepared query parameters with the MCP system."
                )
                
                # Example of how to process a response (would come from MCP tool)
                # response_data = {...}  # This would come from MCP tool execution
                # job_postings = gatherer.process_perplexity_response(response_data)
                # gatherer.insert_jobs_to_database(job_postings)
        
        if args.stats:
            gatherer.show_statistics()
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
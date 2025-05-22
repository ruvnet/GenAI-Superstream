#!/usr/bin/env python3
"""
Main entry point for the advanced DuckDB implementation for UK AI jobs analytics.

This script provides the main functionality to initialize the database, gather data
from PerplexityAI, and perform analytics on UK AI jobs data.
"""

import os
import argparse
import logging
import json
import sys
from typing import Dict, List, Optional, Any, Tuple

from advanced.config import DB_CONFIG, LOG_CONFIG, PERPLEXITY_CONFIG
from advanced.db.connection import connection_manager
from advanced.db.queries import JobsDatabase
from advanced.perplexity.client import PerplexityClient, create_uk_ai_jobs_query
from advanced.perplexity.parsers import perplexity_to_job_postings

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG.get("level", "INFO")),
    format=LOG_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    handlers=[
        logging.FileHandler(LOG_CONFIG.get("log_file")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def init_database() -> JobsDatabase:
    """
    Initialize the database with the proper schema.
    
    Returns:
        JobsDatabase instance
    """
    logger.info("Initializing database...")
    db = JobsDatabase()
    db.initialize_schema()
    logger.info("Database initialization complete")
    return db


def gather_data_from_perplexity(query: Optional[str] = None) -> Dict[str, Any]:
    """
    Prepare to gather data from PerplexityAI.
    
    Args:
        query: Custom query to send to PerplexityAI
        
    Returns:
        Dictionary with parameters for MCP use
    """
    logger.info("Preparing to gather data from PerplexityAI...")
    client = PerplexityClient()
    
    if not query:
        query = create_uk_ai_jobs_query()
    
    mcp_params = client.query_perplexity(query)
    logger.info("PerplexityAI query prepared")
    
    return mcp_params


def process_perplexity_response(response_data: Dict[str, Any], db: JobsDatabase) -> int:
    """
    Process a response from PerplexityAI and insert jobs into the database.
    
    Args:
        response_data: The raw response data from the MCP service
        db: JobsDatabase instance
        
    Returns:
        Number of jobs inserted
    """
    logger.info("Processing PerplexityAI response...")
    client = PerplexityClient()
    
    # Process the response
    perplexity_response = client.process_response(response_data)
    
    # Store the response as a data source
    source_id = db.insert_perplexity_source(perplexity_response)
    
    # Convert to job postings
    job_postings = perplexity_to_job_postings(perplexity_response.content)
    
    # Insert jobs into database
    successful, failed = db.insert_jobs_batch(job_postings)
    
    logger.info(f"Processed {len(job_postings)} job postings: {successful} inserted, {failed} failed")
    return successful


def main():
    """
    Main function to run the advanced DuckDB implementation.
    """
    parser = argparse.ArgumentParser(description="Advanced DuckDB implementation for UK AI jobs analytics")
    parser.add_argument("--init", action="store_true", help="Initialize the database")
    parser.add_argument("--gather", action="store_true", help="Gather data from PerplexityAI")
    parser.add_argument("--query", type=str, help="Custom query for PerplexityAI")
    parser.add_argument("--response-file", type=str, help="JSON file containing PerplexityAI response for processing")
    args = parser.parse_args()
    
    try:
        # Initialize database if requested
        if args.init:
            db = init_database()
            print("Database initialized successfully!")
        else:
            db = JobsDatabase()
        
        # Gather data if requested
        if args.gather:
            mcp_params = gather_data_from_perplexity(args.query)
            print("\nTo gather data, use the following MCP parameters with the use_mcp_tool:")
            print(json.dumps(mcp_params, indent=2))
            print("\nAfter receiving the response, run this script with --response-file=response.json")
        
        # Process response if provided
        if args.response_file:
            if not os.path.exists(args.response_file):
                print(f"Error: Response file '{args.response_file}' not found")
                return
            
            with open(args.response_file, 'r') as f:
                response_data = json.load(f)
            
            jobs_inserted = process_perplexity_response(response_data, db)
            print(f"Successfully inserted {jobs_inserted} jobs into the database")
        
        # If no specific action requested, show help
        if not (args.init or args.gather or args.response_file):
            parser.print_help()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"An error occurred: {e}")
    finally:
        # Close database connection
        connection_manager.close()


def demo():
    """
    Run a demonstration of the advanced DuckDB implementation.
    
    This function provides a simple way to demonstrate the functionality without
    requiring command-line arguments.
    """
    try:
        print("Advanced DuckDB Implementation for UK AI Jobs Analytics Demo")
        print("===========================================================")
        
        # Initialize database
        print("\nInitializing database...")
        db = init_database()
        print("Database initialized successfully!")
        
        # Show how to gather data
        print("\nTo gather data from PerplexityAI, you would use:")
        mcp_params = gather_data_from_perplexity()
        print(f"use_mcp_tool with:\nserver_name: {mcp_params['server_name']}\ntool_name: {mcp_params['tool_name']}")
        
        # Create and insert some sample job postings
        print("\nInserting sample job postings...")
        from advanced.models.data_classes import (
            JobPosting, Skill, Salary, AIImpactLevel, ContractType, SeniorityLevel, SkillCategory
        )
        import datetime
        import uuid
        
        # Create sample job postings
        sample_jobs = [
            JobPosting(
                job_id=f"JOB-{uuid.uuid4().hex[:8]}",
                title="Machine Learning Engineer",
                company="UK Tech Ltd",
                location="London, UK",
                description="We are looking for a Machine Learning Engineer to join our team...",
                date_posted=datetime.date.today(),
                source="sample_data",
                ai_impact=0.85,
                salary_text="£60,000 - £80,000",
                salary=Salary(min_value=60000, max_value=80000, currency="GBP"),
                remote_work=True,
                remote_percentage=50,
                contract_type=ContractType.FULL_TIME,
                seniority_level=SeniorityLevel.SENIOR,
                skills=[
                    Skill(name="Python", category=SkillCategory.TECHNICAL, is_required=True),
                    Skill(name="TensorFlow", category=SkillCategory.AI, is_required=True),
                    Skill(name="PyTorch", category=SkillCategory.AI, is_required=False)
                ]
            ),
            JobPosting(
                job_id=f"JOB-{uuid.uuid4().hex[:8]}",
                title="Data Scientist",
                company="AI Solutions UK",
                location="Manchester, UK (Hybrid)",
                description="Join our data science team to work on cutting-edge AI solutions...",
                date_posted=datetime.date.today() - datetime.timedelta(days=7),
                source="sample_data",
                ai_impact=0.75,
                salary_text="£50,000 - £70,000",
                salary=Salary(min_value=50000, max_value=70000, currency="GBP"),
                remote_work=True,
                remote_percentage=60,
                contract_type=ContractType.FULL_TIME,
                seniority_level=SeniorityLevel.MID_LEVEL,
                skills=[
                    Skill(name="Python", category=SkillCategory.TECHNICAL, is_required=True),
                    Skill(name="SQL", category=SkillCategory.TECHNICAL, is_required=True),
                    Skill(name="Machine Learning", category=SkillCategory.AI, is_required=True)
                ]
            )
        ]
        
        # Insert sample jobs
        successful, failed = db.insert_jobs_batch(sample_jobs)
        print(f"Inserted {successful} sample job postings")
        
        # Demonstrate search functionality
        print("\nSearching for jobs with 'Machine Learning' in the title...")
        results = db.search_jobs(title="Machine Learning")
        print(f"Found {len(results)} matching jobs")
        
        # Show how to retrieve a job
        if results:
            job_id = results[0]['job_id']
            print(f"\nRetrieving job with ID: {job_id}")
            job = db.get_job_by_id(job_id)
            if job:
                print(f"Job Title: {job['title']}")
                print(f"Company: {job['company']}")
                print(f"AI Impact: {job['ai_impact']}")
                print(f"Skills: {', '.join([s['name'] for s in job['skills']])}")
        
        print("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in demo function: {e}")
        print(f"An error occurred: {e}")
    finally:
        # Close database connection
        connection_manager.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # If no arguments provided, run demo
        demo()
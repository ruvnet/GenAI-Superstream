#!/usr/bin/env python3
"""
Example script showing how to integrate the data gatherer with MCP tools.

This demonstrates how to use the data_gatherer module with actual MCP tool calls
to fetch data from PerplexityAI.
"""

import json
import sys
from advanced.data_gatherer import DataGatherer
from advanced.utils.logging import setup_logging

# Set up logging
logger = setup_logging(__name__)


def example_mcp_integration():
    """
    Example of how to integrate the data gatherer with MCP tools.
    
    Note: This is a demonstration script. In actual use, the MCP tool calls
    would be handled by the AI assistant or MCP client system.
    """
    
    # Create data gatherer instance
    gatherer = DataGatherer(dry_run=False)
    
    # Step 1: Initialize the database
    logger.info("Initializing database...")
    gatherer.initialize_database()
    
    # Step 2: Prepare query for PerplexityAI
    logger.info("Preparing query for PerplexityAI...")
    
    # Example 1: Basic query
    job_postings = gatherer.gather_data(
        role="Machine Learning Engineer",
        location="London",
        timeframe="2025"
    )
    
    # The gather_data method returns query parameters that need to be used
    # with the MCP tool. In a real implementation, you would:
    # 1. Get the query parameters from gather_data
    # 2. Use the MCP tool with these parameters
    # 3. Process the response
    
    # Example of what the MCP tool call would look like (pseudo-code):
    """
    <use_mcp_tool>
    <server_name>perplexityai</server_name>
    <tool_name>PERPLEXITYAI_PERPLEXITY_AI_SEARCH</tool_name>
    <arguments>
    {
        "systemContent": "You are a technical data analyst specializing in the UK job market...",
        "userContent": "What are the latest trends in AI technical jobs in the UK? Focus specifically on Machine Learning Engineer roles...",
        "temperature": 0.1,
        "max_tokens": 1000,
        "return_citations": true
    }
    </arguments>
    </use_mcp_tool>
    """
    
    # Step 3: Process the response (example with mock data)
    # In reality, this would come from the MCP tool response
    mock_response = {
        "data": {
            "response": {
                "id": "example-response-id",
                "choices": [{
                    "message": {
                        "content": """
                        Here are some Machine Learning Engineer positions in London:
                        
                        | Job Title | Company | Location | Salary | AI Impact | Skills |
                        |-----------|---------|----------|---------|-----------|---------|
                        | Senior ML Engineer | DeepMind | London | £120k-£180k | 0.95 | Python, TensorFlow, Transformers |
                        | ML Engineer | Meta | London | £100k-£150k | 0.90 | PyTorch, NLP, Computer Vision |
                        | Principal ML Engineer | Google | London | £150k-£200k | 0.98 | JAX, TPUs, Large Language Models |
                        """
                    }
                }],
                "citations": [
                    "https://example.com/job1",
                    "https://example.com/job2"
                ]
            }
        }
    }
    
    logger.info("Processing PerplexityAI response...")
    job_postings = gatherer.process_perplexity_response(mock_response)
    
    # Step 4: Insert jobs into database
    if job_postings:
        logger.info(f"Inserting {len(job_postings)} jobs into database...")
        inserted_count = gatherer.insert_jobs_to_database(job_postings)
        logger.info(f"Successfully inserted {inserted_count} jobs")
    
    # Step 5: Show statistics
    logger.info("\nDatabase statistics after insertion:")
    gatherer.show_statistics()


def example_custom_query():
    """Example of using a custom query."""
    gatherer = DataGatherer(dry_run=False)
    
    # Custom query example
    custom_query = """
    Find the most innovative AI startups in the UK that are hiring for AI researchers 
    and engineers. Focus on companies working on AGI, robotics, or healthcare AI. 
    Include information about their funding, team size, and the specific AI problems 
    they're solving. Format the response as a detailed table with all job information.
    """
    
    # This would prepare the query for MCP tool use
    gatherer.gather_data(query=custom_query)
    
    logger.info("Custom query prepared for PerplexityAI MCP tool")


def example_batch_processing():
    """Example of batch processing multiple queries."""
    gatherer = DataGatherer(dry_run=False)
    
    # Different roles to query
    roles = [
        "Machine Learning Engineer",
        "AI Research Scientist", 
        "MLOps Engineer",
        "Computer Vision Engineer",
        "NLP Engineer"
    ]
    
    # Different locations
    locations = ["London", "Manchester", "Edinburgh", "Cambridge", "Bristol"]
    
    logger.info("Preparing batch queries...")
    
    for role in roles:
        for location in locations:
            logger.info(f"\nPreparing query for {role} in {location}")
            gatherer.gather_data(
                role=role,
                location=location,
                timeframe="2025",
                batch_size=5  # Smaller batch for each specific query
            )
            
            # In a real implementation, you would:
            # 1. Execute the MCP tool call
            # 2. Process the response
            # 3. Insert into database
            # 4. Add a delay to avoid rate limiting


if __name__ == "__main__":
    print("""
    PerplexityAI MCP Integration Examples
    =====================================
    
    This script demonstrates how to use the data_gatherer module with MCP tools.
    
    Note: Actual MCP tool execution requires integration with the AI assistant
    or MCP client system. This script shows the preparation steps and how to
    process responses.
    
    """)
    
    try:
        # Run the main example
        example_mcp_integration()
        
        # Show other examples (commented out to avoid multiple executions)
        # example_custom_query()
        # example_batch_processing()
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
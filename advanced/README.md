# Advanced DuckDB Implementation for UK AI Jobs Analytics

This advanced implementation provides enhanced functionality for analyzing UK AI jobs data with DuckDB, featuring PerplexityAI MCP integration for data gathering.

## Overview

This implementation extends the basic DuckDB implementation with:

- Advanced schema design with more detailed job metadata
- Enhanced analytics capabilities using scikit-learn
- Data visualization for insights and trend analysis
- PerplexityAI MCP integration for real-time job data gathering
- Improved error handling and performance optimization
- Modular and extensible architecture

## File Structure

```
advanced/
├── __init__.py                # Package initialization
├── config.py                  # Configuration settings
├── data_gatherer.py           # CLI tool for data gathering
├── main.py                    # Main entry point
├── example_mcp_integration.py # MCP integration examples
├── init_duckdb.py             # Legacy database initialization
├── .env.sample                # Environment configuration template
├── README.md                  # This documentation
├── README_data_gatherer.md    # Data gatherer documentation
├── models/                    # Data models and schemas
│   ├── __init__.py
│   ├── data_classes.py        # Data class definitions
│   └── schemas.py             # Database schema definitions
├── db/                        # Database operations
│   ├── __init__.py
│   ├── connection.py          # Database connection manager
│   └── queries.py             # SQL query operations
├── analytics/                 # Analytics functionality
│   ├── __init__.py
│   ├── metrics.py             # Analytics metrics calculation
│   └── visualizations.py      # Data visualization functions
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── helpers.py             # Helper functions
│   └── logging.py             # Logging configuration
├── perplexity/                # PerplexityAI integration
│   ├── __init__.py
│   ├── client.py              # PerplexityAI MCP client
│   ├── data_processor.py      # Data processing utilities
│   └── parsers.py             # Response parsing utilities
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Test configuration
│   ├── test_client.py         # Client tests
│   ├── test_config.py         # Configuration tests
│   ├── test_data_gatherer.py  # Data gatherer tests
│   ├── test_data_processor.py # Data processor tests
│   └── FIXES_DOCUMENTATION.md # Test fixes documentation
├── logs/                      # Log files directory
├── exports/                   # Export files directory
└── visualizations/            # Generated visualizations directory
```

## Key Features

### Enhanced Database Schema

The advanced implementation uses an extended schema that captures more detailed information about job postings:

- Structured salary information (min/max/currency)
- Remote work options and percentages
- Detailed AI impact metrics and categorization
- Required and preferred skills with categories
- Contract types and seniority levels
- Historical data tracking for trend analysis

### PerplexityAI MCP Integration

This implementation leverages the PerplexityAI MCP (Model Context Protocol) to gather real-time data about AI jobs in the UK market. The integration:

1. Sends structured queries to PerplexityAI asking for specific job data
2. Parses the responses to extract job information
3. Transforms the data into structured job postings
4. Stores the data in the DuckDB database for analysis

The PerplexityAI integration provides access to up-to-date information about:
- Current job titles and roles in AI
- Salary ranges for different positions
- Required skills and qualifications
- AI impact across different sectors
- Trends in the UK job market

### Advanced Analytics

The implementation includes sophisticated analytics capabilities:

- Cluster analysis to identify patterns in job data
- Skills importance analysis to track in-demand skills
- Salary trend analysis based on AI impact
- Location and remote work analysis
- Time-series analysis for market trends
- Visualizations to communicate insights effectively

## Installation

### Prerequisites

- Python 3.8+
- Required packages: duckdb, pandas, numpy, scikit-learn, matplotlib, seaborn, python-dotenv

### Install Required Packages

```bash
pip install duckdb pandas numpy scikit-learn matplotlib seaborn python-dotenv
```

### Configuration

Copy the environment configuration template and customize as needed:

```bash
cp advanced/.env.sample advanced/.env
```

Edit `advanced/.env` to configure:

```env
# Database Configuration
DB_PATH=./duckdb_advanced.db
DB_MEMORY_LIMIT=4GB
DB_THREADS=4
DB_CACHE_ENABLED=True
DB_CACHE_SIZE=100

# Logging Configuration
LOG_FILE=./logs/jobs_db.log
LOG_LEVEL=INFO

# PerplexityAI MCP Configuration
PERPLEXITY_MCP_SERVER_NAME=perplexityai
PERPLEXITY_MCP_URL=http://localhost:3001
```

## Usage

### Initializing the Database

```python
from advanced.main import init_database

# Initialize the database with the enhanced schema
db = init_database()
print("Database initialized successfully!")
```

### Gathering Data from PerplexityAI

To gather data from PerplexityAI, you need to use the MCP service:

```python
from advanced.main import gather_data_from_perplexity
import json

# Prepare the query parameters
mcp_params = gather_data_from_perplexity()

# The parameters can be used with the use_mcp_tool
print("To gather data, use the following MCP parameters:")
print(json.dumps(mcp_params, indent=2))

# After receiving the response, save it to a file (response.json)
# Then process it:
from advanced.main import process_perplexity_response

with open('response.json', 'r') as f:
    response_data = json.load(f)

jobs_inserted = process_perplexity_response(response_data, db)
print(f"Successfully inserted {jobs_inserted} jobs into the database")
```

### Example PerplexityAI Query

```
What are the latest trends in AI technical jobs in the UK? Focus specifically on Machine Learning Engineer roles.
Please provide comprehensive data on job titles, companies, locations, salary ranges, job descriptions, 
AI impact metrics, posting dates, and data sources. Structure your response as a table for database insertion.
```

### Running Analytics

```python
from advanced.db.queries import JobsDatabase
from advanced.analytics.metrics import (
    calculate_ai_impact_distribution,
    calculate_top_companies,
    calculate_skills_importance,
    perform_cluster_analysis
)

# Initialize database connection
db = JobsDatabase()

# Get AI impact distribution
impact_df = calculate_ai_impact_distribution(db)
print("AI Impact Distribution:")
print(impact_df)

# Get top companies by AI job count
companies_df = calculate_top_companies(db, min_ai_impact=0.6, limit=10)
print("\nTop Companies by AI Jobs:")
print(companies_df)

# Get skills importance
skills_df = calculate_skills_importance(db, top_n=20)
print("\nTop Skills:")
print(skills_df)

# Perform cluster analysis
df_with_clusters, cluster_stats = perform_cluster_analysis(db, n_clusters=4)
print("\nCluster Statistics:")
print(cluster_stats)
```

### Creating Visualizations

```python
from advanced.analytics.visualizations import create_visualizations

# Create all visualizations
viz_paths = create_visualizations(db)
print(f"Created {len(viz_paths)} visualizations:")
for path in viz_paths:
    print(f"- {path}")
```

### Using the Command-Line Interface

The implementation includes multiple command-line interfaces for different operations:

#### Main Entry Point

```bash
# Initialize the database
python advanced/main.py --init

# Prepare a query for PerplexityAI
python advanced/main.py --gather

# Process a PerplexityAI response
python advanced/main.py --response-file=response.json

# Run a demonstration with sample data
python advanced/main.py
```

#### Data Gatherer CLI

```bash
# Initialize database with enhanced schema
python advanced/data_gatherer.py --init

# Gather data with default query
python advanced/data_gatherer.py --gather

# Gather data with specific role and location
python advanced/data_gatherer.py --gather --role "Machine Learning Engineer" --location "London"

# Gather data with custom query
python advanced/data_gatherer.py --gather --query "What are the highest paying AI jobs in Manchester?"

# View database statistics
python advanced/data_gatherer.py --stats

# Dry run mode (preview without execution)
python advanced/data_gatherer.py --gather --dry-run
```

## PerplexityAI MCP Data Gathering

The PerplexityAI MCP integration is a key feature of this advanced implementation. It allows you to gather real-time data about AI jobs in the UK market.

### How it Works

1. **Query Preparation**: The system formulates a detailed query asking about AI jobs in the UK, specifying the need for structured data.

2. **MCP Tool Usage**: The query is sent to PerplexityAI using the MCP tool:

```python
mcp_params = {
    "server_name": "perplexityai",
    "tool_name": "PERPLEXITYAI_PERPLEXITY_AI_SEARCH",
    "arguments": {
        "systemContent": "You are a technical data analyst specializing in the UK job market...",
        "userContent": "What are the latest trends in AI technical jobs in the UK?...",
        "temperature": 0.1,
        "max_tokens": 1000,
        "return_citations": True
    }
}
```

3. **Response Processing**: The response from PerplexityAI is parsed to extract structured job data:
   - Table extraction from markdown format
   - Job information parsing (title, company, location, salary, etc.)
   - AI impact estimation based on job content
   - Skills extraction from descriptions
   - Data transformation into JobPosting objects

4. **Database Insertion**: The extracted job postings are inserted into the DuckDB database.

5. **Source Tracking**: The PerplexityAI response is also stored as a data source for reference.

### Example Usage

To gather data using PerplexityAI MCP:

1. **Prepare the query**:

```python
from advanced.main import gather_data_from_perplexity

# Get MCP parameters
mcp_params = gather_data_from_perplexity("What are the latest trends in AI technical jobs in London?")
```

2. **Use the MCP tool** (This step is performed using the UI):

```
<use_mcp_tool>
<server_name>perplexityai</server_name>
<tool_name>PERPLEXITYAI_PERPLEXITY_AI_SEARCH</tool_name>
<arguments>
{
  "systemContent": "You are a technical data analyst specializing in the UK job market...",
  "userContent": "What are the latest trends in AI technical jobs in London?...",
  "temperature": 0.1,
  "max_tokens": 1000,
  "return_citations": true
}
</arguments>
</use_mcp_tool>
```

3. **Process the response**:

```python
from advanced.main import process_perplexity_response
from advanced.db.queries import JobsDatabase

# Initialize database
db = JobsDatabase()

# Process the response (assuming the response is stored in response_data)
jobs_inserted = process_perplexity_response(response_data, db)
```

## Advanced Usage Examples

### Custom Data Analysis

```python
from advanced.db.queries import JobsDatabase

# Initialize database
db = JobsDatabase()

# Custom query - AI impact by location
location_impact_df = db.to_dataframe("""
    SELECT 
        CASE 
            WHEN location LIKE '%London%' THEN 'London'
            WHEN location LIKE '%Manchester%' THEN 'Manchester'
            WHEN location LIKE '%Remote%' THEN 'Remote'
            ELSE 'Other'
        END as location_group,
        COUNT(*) as job_count,
        AVG(ai_impact) as avg_ai_impact,
        AVG(salary_min) as avg_min_salary
    FROM job_postings
    GROUP BY location_group
    ORDER BY avg_ai_impact DESC
""")

print("AI Impact by Location:")
print(location_impact_df)
```

### Custom Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns
from advanced.db.queries import JobsDatabase
from advanced.analytics.metrics import calculate_skills_importance

# Initialize database
db = JobsDatabase()

# Get skills data
skills_df = calculate_skills_importance(db, top_n=15)

# Create a custom visualization
plt.figure(figsize=(12, 8))
sns.barplot(x='job_count', y='skill_name', data=skills_df, hue='skill_category')
plt.title('Top Skills by Job Count')
plt.xlabel('Number of Jobs')
plt.ylabel('Skill')
plt.tight_layout()
plt.savefig('top_skills.png')
```

## Extending the Implementation

The modular architecture makes it easy to extend the implementation:

1. **Adding new analytics**: Create new functions in the `analytics` module.
2. **Adding new data sources**: Extend the `perplexity` module or create new modules for other data sources.
3. **Adding new schema elements**: Update the schema definitions in `models/schemas.py`.
4. **Adding new visualizations**: Create new visualization functions in `analytics/visualizations.py`.

Example of adding a new analytics function:

```python
# In analytics/metrics.py
def calculate_remote_vs_office_comparison(db: JobsDatabase) -> pd.DataFrame:
    """
    Compare remote vs. office-based jobs by AI impact and salary.
    
    Args:
        db: JobsDatabase instance
        
    Returns:
        DataFrame with comparison data
    """
    query = """
    SELECT 
        CASE WHEN remote_work = TRUE THEN 'Remote' ELSE 'Office' END as work_type,
        COUNT(*) as job_count,
        AVG(ai_impact) as avg_ai_impact,
        AVG(salary_min) as avg_min_salary,
        AVG(salary_max) as avg_max_salary
    FROM job_postings
    GROUP BY work_type
    """
    return db.to_dataframe(query)
```

## Testing

The advanced implementation includes a comprehensive test suite to ensure reliability and correctness:

```bash
# Run all tests
python -m pytest advanced/tests/

# Run specific test files
python -m pytest advanced/tests/test_client.py
python -m pytest advanced/tests/test_data_gatherer.py

# Run with coverage
python -m pytest advanced/tests/ --cov=advanced

# Run with verbose output
python -m pytest advanced/tests/ -v
```

### Test Coverage

The test suite covers:

- **Configuration Management**: Environment variable loading and validation
- **Database Operations**: Connection management, schema creation, data insertion
- **PerplexityAI Integration**: Client functionality and response parsing
- **Data Processing**: Job data transformation and validation
- **Data Gatherer CLI**: Command-line interface functionality
- **Error Handling**: Various failure scenarios and edge cases

### Test Files

- [`test_config.py`](tests/test_config.py): Configuration system tests
- [`test_client.py`](tests/test_client.py): PerplexityAI client tests
- [`test_data_processor.py`](tests/test_data_processor.py): Data processing tests
- [`test_data_gatherer.py`](tests/test_data_gatherer.py): CLI tool tests
- [`conftest.py`](tests/conftest.py): Shared test fixtures and configuration

## Future Enhancements

Potential future enhancements to this implementation:

1. **Real-time monitoring**: Track job market changes over time
2. **Advanced ML models**: Predictive analytics for salary and job growth
3. **Interactive dashboards**: Web-based visualizations using Plotly or Dash
4. **Integration with more data sources**: Combine data from multiple sources
5. **Natural language querying**: Allow users to ask questions in natural language
6. **Automated reporting**: Generate periodic reports on job market trends
7. **API endpoints**: REST API for accessing job data and analytics
8. **Automated data quality monitoring**: Continuous validation of data integrity

## Conclusion

This advanced DuckDB implementation provides a powerful framework for analyzing UK AI jobs data, with PerplexityAI MCP integration for data gathering. The modular architecture makes it easy to extend and customize the implementation for specific use cases.

The combination of DuckDB's analytical capabilities, scikit-learn's machine learning algorithms, and PerplexityAI's data gathering capabilities enables sophisticated analysis of the UK AI job market, helping users understand trends, identify in-demand skills, and track the impact of AI on technical roles.
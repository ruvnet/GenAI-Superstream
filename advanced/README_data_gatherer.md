# Data Gatherer for PerplexityAI MCP Integration

## Overview

The `data_gatherer.py` script provides a command-line interface for gathering AI job data from PerplexityAI using the MCP (Model Context Protocol) integration and storing it in a DuckDB database.

## Features

- **Database Initialization**: Set up required tables and indexes for storing job data
- **Data Gathering**: Query PerplexityAI for UK AI job market data
- **Flexible Queries**: Support for custom queries, specific roles, locations, and timeframes
- **Batch Processing**: Process jobs in configurable batch sizes
- **Data Processing**: Transform and enrich job data before storage
- **Statistics**: View database statistics and insights
- **Dry Run Mode**: Preview operations without making changes

## Installation

Ensure you have the required dependencies installed:

```bash
pip install duckdb python-dotenv
```

## Configuration

The script uses configuration from `advanced/config.py` and environment variables from `advanced/.env`:

```env
# PerplexityAI MCP Configuration
PERPLEXITY_MCP_SERVER_NAME=perplexityai
PERPLEXITY_MCP_URL=http://localhost:3001

# Database Configuration
DB_PATH=./duckdb_advanced.db
DB_MEMORY_LIMIT=4GB
DB_THREADS=4
```

## Usage

### Initialize Database

Create the required tables and indexes:

```bash
python advanced/data_gatherer.py --init
```

### Gather Data

#### Basic Usage
```bash
python advanced/data_gatherer.py --gather
```

#### With Specific Role and Location
```bash
python advanced/data_gatherer.py --gather --role "Machine Learning Engineer" --location "London"
```

#### With Custom Query
```bash
python advanced/data_gatherer.py --gather --query "What are the highest paying AI jobs in Manchester?"
```

#### With Timeframe
```bash
python advanced/data_gatherer.py --gather --timeframe "last 3 months"
```

### View Statistics

Show current database statistics:

```bash
python advanced/data_gatherer.py --stats
```

### Dry Run Mode

Preview what would be done without executing:

```bash
python advanced/data_gatherer.py --gather --dry-run
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--init` | Initialize the database with required tables and indexes |
| `--gather` | Gather data from PerplexityAI |
| `--stats` | Show statistics about the current database content |
| `--query` | Custom query string for PerplexityAI |
| `--role` | Specific AI role to focus on (e.g., "Machine Learning Engineer") |
| `--location` | Specific UK location (e.g., "London") |
| `--timeframe` | Timeframe for the data (e.g., "last 3 months") |
| `--batch-size` | Number of jobs to process in each batch (default: 10) |
| `--dry-run` | Show what would be done without actually executing |

## Database Schema

The script creates the following tables:

### job_postings
- Core job information including title, company, location, salary
- AI impact metrics and categorization
- Remote work options and contract types
- Timestamps for tracking

### job_skills
- Skills required for each job
- Skill categories (AI, Technical, etc.)
- Required vs. preferred indicators

### ai_impact_metrics
- Detailed AI impact analysis
- Automation risk scores
- Augmentation potential
- Transformation levels

### Additional Tables
- `companies`: Company information and AI focus levels
- `job_history`: Track changes to job postings
- `perplexity_data_sources`: Track data sources and queries
- `data_quality_metrics`: Monitor data quality

## Integration with MCP Tools

The data gatherer prepares queries for the PerplexityAI MCP tool but doesn't execute them directly. In a production environment, you would:

1. Use the data gatherer to prepare query parameters
2. Execute the MCP tool call using the AI assistant or MCP client
3. Process the response using the data gatherer's processing methods
4. Insert the processed data into the database

See `example_mcp_integration.py` for a complete example of the integration workflow.

## Example Workflow

```python
from advanced.data_gatherer import DataGatherer

# Create gatherer instance
gatherer = DataGatherer()

# Initialize database
gatherer.initialize_database()

# Prepare query (returns parameters for MCP tool)
query_params = gatherer.gather_data(
    role="AI Research Scientist",
    location="Cambridge",
    timeframe="2025"
)

# Execute MCP tool with query_params (handled externally)
# response = execute_mcp_tool(query_params)

# Process response
job_postings = gatherer.process_perplexity_response(response)

# Insert into database
gatherer.insert_jobs_to_database(job_postings)

# View statistics
gatherer.show_statistics()
```

## Logging

The script uses comprehensive logging to track operations:
- All operations are logged with timing information
- Errors are logged with full context
- Log files are stored in `advanced/logs/`

## Error Handling

- Database connection errors are caught and logged
- Failed job insertions don't stop the batch
- Graceful handling of malformed data
- Keyboard interrupts are handled cleanly

## Performance Considerations

- Batch processing to avoid overwhelming the API
- Database indexes for fast querying
- Connection pooling for database operations
- Query caching for repeated operations

## Future Enhancements

- [ ] Automatic scheduling for periodic data updates
- [ ] Export functionality for analysis results
- [ ] Integration with job boards APIs
- [ ] Machine learning models for job matching
- [ ] Real-time monitoring dashboard
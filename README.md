# GenAI-Superstream

> Agentic Engineering for Data Analysis

## Table of Contents

- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
  - [Agentics](#agentics)
  - [Vibe Coding](#vibe-coding)
  - [SPARC Methodology](#sparc-methodology)
  - [Model Context Protocol (MCP)](#model-context-protocol-mcp)
- [Project Implementation](#project-implementation)
  - [DuckDB Overview](#duckdb-overview)
  - [UK AI Jobs Data Analysis](#uk-ai-jobs-data-analysis)
  - [Scikit-learn Integration](#scikit-learn-integration)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Tutorials](#tutorials)
  - [Setting Up the DuckDB Database](#setting-up-the-duckdb-database)
  - [Populating with Jobs Data](#populating-with-jobs-data)
  - [Running Analytics](#running-analytics)
  - [Using the CLI Tool](#using-the-cli-tool)
- [Advanced Usage](#advanced-usage)
  - [Custom Analytics](#custom-analytics)
  - [Extending the Database](#extending-the-database)
  - [Integration with Other Systems](#integration-with-other-systems)
- [Advanced Implementation](#advanced-implementation)
  - [PerplexityAI MCP Integration](#perplexityai-mcp-integration)
  - [Enhanced Schema and Analytics](#enhanced-schema-and-analytics)
  - [Command-Line Tools](#command-line-tools)
- [MCP Server](#mcp-server)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

GenAI-Superstream is a project demonstrating Agentic Engineering and "Vibe Coding" principles for Data Science applications. Created by Reuven Cohen (rUv), this project showcases the integration of advanced AI-powered data analysis techniques with a focus on analyzing the impact of AI on technical jobs in the UK job market.

This implementation demonstrates how to use DuckDB (a high-performance analytical database) in combination with scikit-learn to collect, analyze, and visualize jobs data, with a particular focus on understanding how AI is transforming technical roles in the UK job market.

## Core Concepts

### Agentics

Agentics (pronounced /əˈdʒentɪks/) is the field of study and engineering practice focused on creating intelligent systems that are:

- **P**roactive: Anticipating and initiating changes
- **A**utonomous: Operating independently
- **C**ollaborative: Working effectively with other systems or agents
- **T**argeted: Pursuing defined objectives

This PACT framework guides the development of AI systems that can work alongside humans to solve complex problems.

### Vibe Coding

Vibe Coding is an approach to software development that emphasizes:

- Crafting code by feel, balancing functionality with readability, style and project mood
- Rapid iteration guided by intuitive feedback
- Emphasis on clean, expressive naming and structure
- Aligning code style with team culture and project "vibe"

Benefits include:
- Faster prototyping with fewer mental blocks
- Code that feels approachable and motivates collaboration
- Consistency through shared aesthetic standards

### Vibe Coding vs Agentic Engineering

| Vibe Coding | Agentic Engineering |
|-------------|---------------------|
| Flow | Structured |
| Fluid and Intuitive | Process Driven |
| Little Process | Deep Planning / Test Driven |
| Human is the feedback loop | Machine is feedback loop |
| Ideation and Discovery | Iteration and refinement |

### SPARC Methodology

SPARC is a comprehensive methodology designed to guide the development of robust and scalable applications. SPARC stands for:

- **Specification**: Define clear objectives, detailed requirements, user scenarios, and UI/UX standards
- **Pseudocode**: Map out logical implementation pathways before coding
- **Architecture**: Design modular, maintainable system components using appropriate technology stacks
- **Refinement**: Iteratively optimize code using autonomous feedback loops and stakeholder inputs
- **Completion**: Conduct rigorous testing, finalize comprehensive documentation, and deploy structured monitoring strategies

Each step ensures thorough planning, execution, and reflection throughout the project lifecycle.

### Model Context Protocol (MCP)

MCP (Model Context Protocol) is the new standard for LLM-tool integration:

- Simple, composable, and totally abstracted
- Turns any tool into a native function any model can call
- Secure, two-way connections between models and external tools
- Plug-and-play "USB-C for AI"
- Zero-friction developer experience
- Unix mindset applied to AI
- Built-in guardrails

## Project Implementation

### DuckDB Overview

DuckDB is a high-performance, in-process analytical database management system optimized for complex, large-scale analytical SQL queries. It is often described as "SQLite for analytics" due to its lightweight, easy integration and ability to run within the same process as the application without the need for a separate database server.

Key features:
- **Columnar-vectorized query execution engine**: Processes large batches of data at once
- **Full SQL support**: Including complex queries and window functions
- **Support for popular data formats**: CSV, Parquet, etc.
- **ACID transactional guarantees**: Ensures data integrity
- **Seamless Python integration**: Works with pandas DataFrames

### UK AI Jobs Data Analysis

This project uses DuckDB to analyze AI's impact on technical jobs in the UK. Key components:

1. **Data Collection**: Gathering representative job postings from reliable sources
2. **Data Storage**: Structured database schema optimized for analytics
3. **Data Analysis**: SQL queries and scikit-learn integration for insights
4. **Visualization**: Visual representation of trends and patterns

The job data includes information such as:
- Job titles and companies
- Locations and salary ranges
- Job descriptions
- Metrics quantifying AI's impact on each role
- Posting dates and sources

### Scikit-learn Integration

The project demonstrates how to integrate DuckDB with scikit-learn for advanced analytics:

1. **Feature Engineering**: Extracting relevant features from job data
2. **Clustering**: Identifying patterns and grouping similar jobs
3. **Trend Analysis**: Tracking changes in AI impact over time
4. **Visualization**: Creating insightful visualizations of the data

## Getting Started

### Prerequisites

- Python 3.8+ (3.6+ for basic implementation)
- pip (Python package manager)

### Dependencies

The project uses different dependency sets depending on the implementation:

**Basic Implementation (`db/` directory):**
- `duckdb` - High-performance analytical database
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning library
- `matplotlib` - Plotting and visualization
- `tabulate` - Pretty-print tabular data

**Advanced Implementation (`advanced/` directory):**
- All basic dependencies plus:
- `seaborn` - Statistical data visualization
- `python-dotenv` - Environment variable management

**MCP Server (`genai-mcp/` directory):**
- `mcp` - Model Context Protocol framework
- `gradio` - Web-based UI framework
- `pytest` - Testing framework
- `pyyaml` - YAML configuration support

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ruvnet/GenAI-Superstream.git
   cd GenAI-Superstream
   ```

2. Install the required dependencies:

   **Basic Implementation:**
   ```bash
   pip install duckdb pandas numpy scikit-learn matplotlib tabulate
   ```

   **Advanced Implementation (recommended):**
   ```bash
   pip install duckdb pandas numpy scikit-learn matplotlib seaborn python-dotenv
   ```

3. Set up the MCP server (optional):
   ```bash
   cd genai-mcp
   make setup
   ```

4. Optional: Install the Roo Code extension for VS Code:
   - Open VS Code
   - Go to Extensions view
   - Search for Roo Code
   - Click Install

### Usage

The project consists of several Python scripts that demonstrate different aspects of the system:

- `db/init_duckdb.py`: Core database initialization and utility functions
- `db/jobs_analytics_example.py`: Example analytics and visualization with scikit-learn
- `db/insert_ai_jobs_duckdb.py`: Script to insert representative UK AI jobs data
- `scripts/review_uk_jobs.py`: CLI tool to review and filter job data

## Tutorials

### Setting Up the DuckDB Database

To initialize the DuckDB database with the proper schema:

```python
from db.init_duckdb import JobsDatabase

# Create a new database instance
db = JobsDatabase()

# Initialize the schema
db.initialize_schema()

print("Database initialized successfully!")
db.close()
```

### Populating with Jobs Data

You can populate the database with UK AI jobs data using the PerplexityAI MCP integration in Roo Code. This approach leverages AI to gather the latest jobs data:

#### Using PerplexityAI MCP

1. First, configure the Composio PerplexityAI MCP in your project by creating an `mcp.json` file:

```json
{
  "servers": {
    "perplexityai": {
      "url": "mcp.composio/your-key-url",
      "tools": [
        "PERPLEXITYAI_PERPLEXITY_AI_SEARCH"
      ]
    }
  }
}
```

2. In Roo Code, interact with the PerplexityAI service to research UK AI jobs data. Ask for structured information about how AI is affecting technical jobs in the UK, specifying that you need job titles, companies, locations, salaries, descriptions, AI impact metrics, posting dates, and sources. Request the data in a tabular format suitable for database ingestion.

When crafting your query, use system instructions that request concise, technical responses with structured data formats. For the user content, specifically ask about the latest trends and data on AI's impact on technical jobs in the UK, emphasizing that you need complete job posting information with all required fields for your database.

The PerplexityAI service will return comprehensive, structured information about current AI-related technical jobs in the UK job market, which you can then parse and insert into your DuckDB database.

2. Parse the returned data and insert it into the DuckDB database:

```python
import duckdb
import json

# Parse data from PerplexityAI response
perplexity_response = json.loads(perplexity_result)
jobs_data = extract_jobs_from_response(perplexity_response)

# Connect to DuckDB and insert data
con = duckdb.connect('db/uk_jobs.duckdb')
for job in jobs_data:
    con.execute('''
        INSERT OR REPLACE INTO jobs
        (job_id, title, company, location, salary, description, ai_impact, date_posted, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        job["job_id"],
        job["title"],
        job["company"],
        job["location"],
        job["salary"],
        job["description"],
        job["ai_impact"],
        job["date_posted"],
        job["source"]
    ))
con.close()
```

3. Alternatively, use the provided script for sample data:

```bash
python db/insert_ai_jobs_duckdb.py
```

This approach demonstrates how to use Roo Code's MCP capabilities to collect real-time data about AI's impact on the UK job market, directly feeding it into your analytics pipeline.

### Running Analytics

To run basic analytics and visualizations on the jobs data:

```bash
python db/jobs_analytics_example.py
```

This script will:
1. Load data from the DuckDB database
2. Preprocess the data for machine learning
3. Run simple clustering and trend analysis on AI impact
4. Generate visualizations showing the results

The visualization will be saved as `db/job_analytics_results.png`.

### Using the CLI Tool

The project includes a command-line tool for quickly reviewing and filtering job data:

```bash
# View all jobs
python scripts/review_uk_jobs.py

# Filter by job title
python scripts/review_uk_jobs.py --title Engineer

# Filter by company
python scripts/review_uk_jobs.py --company Google
```

## Advanced Usage

### Custom Analytics

You can create custom analytics by combining DuckDB's SQL capabilities with scikit-learn:

```python
from db.init_duckdb import JobsDatabase
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Connect to the database
db = JobsDatabase()

# Query specific data
df = db.to_dataframe("SELECT * FROM job_postings WHERE ai_impact > 0.7")

# Create text features from job descriptions
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
features = tfidf.fit_transform(df['description'])

# Run clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(features)

# Analyze results
cluster_stats = df.groupby('cluster').agg({
    'ai_impact': ['mean', 'count'],
    'title': lambda x: ', '.join(set(x))[:100]
})

print(cluster_stats)
db.close()
```

### Extending the Database

You can extend the database schema to include additional information:

```python
# Connect to DuckDB
import duckdb
con = duckdb.connect('db/uk_jobs.duckdb')

# Add a new column
con.execute('''
    ALTER TABLE jobs ADD COLUMN remote_work BOOLEAN;
''')

# Update existing records
con.execute('''
    UPDATE jobs 
    SET remote_work = (location LIKE '%Remote%')
''')

con.close()
```

### Integration with Other Systems

The DuckDB database can be integrated with other systems:

```python
# Export to CSV
import duckdb
con = duckdb.connect('db/uk_jobs.duckdb')
con.execute('''
    COPY (SELECT * FROM jobs) TO 'exported_jobs.csv' (HEADER, DELIMITER ',');
''')
con.close()

# Export to a different database system
import duckdb
import pandas as pd
import sqlite3

# Extract from DuckDB
con_duck = duckdb.connect('db/uk_jobs.duckdb')
df = con_duck.execute("SELECT * FROM jobs").fetch_df()
con_duck.close()

# Load into SQLite
con_sqlite = sqlite3.connect('other_system.db')
df.to_sql('jobs', con_sqlite, if_exists='replace', index=False)
con_sqlite.close()
```

## Advanced Implementation

The `advanced/` directory contains an enhanced implementation of the GenAI-Superstream project with advanced features including PerplexityAI MCP integration, enhanced database schemas, and sophisticated analytics capabilities.

### PerplexityAI MCP Integration

The advanced implementation includes integration with PerplexityAI through the Model Context Protocol (MCP), enabling real-time data gathering from AI-powered search:

```bash
# Initialize the advanced database
python advanced/main.py --init

# Gather data from PerplexityAI (prepares MCP query)
python advanced/main.py --gather

# Process PerplexityAI response
python advanced/main.py --response-file=response.json
```

The PerplexityAI integration allows you to:
- Query for the latest UK AI job market trends
- Extract structured job data from search results
- Automatically parse and store job information
- Track data sources and maintain quality metrics

### Enhanced Schema and Analytics

The advanced implementation features:

- **Comprehensive Job Schema**: Enhanced job postings table with detailed fields for salary ranges, remote work options, AI impact metrics, and skills tracking
- **Skills Analysis**: Separate skills table for tracking required vs. preferred skills with categories
- **Company Tracking**: Company information with AI focus levels
- **Historical Data**: Job history tracking for trend analysis
- **Advanced Analytics**: Clustering, skills importance analysis, salary trends, and visualization capabilities

Example analytics usage:

```python
from advanced.analytics.metrics import calculate_ai_impact_distribution, perform_cluster_analysis
from advanced.db.queries import JobsDatabase

db = JobsDatabase()

# Get AI impact distribution
impact_df = calculate_ai_impact_distribution(db)

# Perform cluster analysis
df_with_clusters, cluster_stats = perform_cluster_analysis(db, n_clusters=4)
```

### Command-Line Tools

The advanced implementation provides comprehensive CLI tools:

```bash
# Data gathering with flexible options
python advanced/data_gatherer.py --gather --role "Machine Learning Engineer" --location "London"

# View database statistics
python advanced/data_gatherer.py --stats

# Dry run mode to preview operations
python advanced/data_gatherer.py --gather --dry-run
```

See [`advanced/README.md`](advanced/README.md) for detailed documentation of the advanced features.

## MCP Server

The project includes a complete MCP (Model Context Protocol) server implementation in the `genai-mcp/` directory:

- **Tools**: Functions for data analysis and job market queries
- **Resources**: Access to job data and analytics results
- **Prompts**: Templates for AI-powered job market analysis
- **Server-Sent Events**: Real-time updates and monitoring

To run the MCP server:

```bash
cd genai-mcp
make setup
make dev
```

The MCP server enables seamless integration with AI assistants and other MCP-compatible applications. See [`genai-mcp/README.md`](genai-mcp/README.md) for complete setup and usage instructions.

## Project Structure

```
GenAI-Superstream/
├── advanced/                    # Advanced implementation with MCP integration
│   ├── analytics/              # Advanced analytics and visualizations
│   ├── db/                     # Database operations and queries
│   ├── models/                 # Data models and database schemas
│   ├── perplexity/             # PerplexityAI MCP integration
│   ├── tests/                  # Comprehensive test suite
│   ├── utils/                  # Utility functions and logging
│   ├── data_gatherer.py        # CLI tool for data gathering
│   ├── main.py                 # Main entry point
│   └── README.md               # Advanced implementation documentation
├── db/                         # Basic DuckDB implementation
│   ├── init_duckdb.py          # Core database class and utilities
│   ├── jobs_analytics_example.py # Example analytics and visualization
│   ├── insert_ai_jobs_duckdb.py # Script to insert representative data
│   ├── README.md               # Database documentation
│   └── uk_jobs.duckdb          # The DuckDB database file
├── genai-mcp/                  # MCP server implementation
│   ├── genai_mcp/              # Server source code
│   ├── sse_server.py           # Server-sent events server
│   └── README.md               # MCP server documentation
├── docs/                       # Project documentation
├── presentation/               # Presentation materials
├── scripts/                    # Utility scripts
└── README.md                   # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Created by Reuven Cohen (rUv) - https://github.com/ruvnet/GenAI-Superstream
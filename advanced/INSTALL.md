# Advanced DuckDB Installation Guide

This guide provides comprehensive instructions for installing and setting up the Advanced DuckDB implementation for UK AI jobs analytics.

## Quick Start

### Option 1: Bash Script (Linux/macOS)
```bash
cd advanced
./install.sh
```

### Option 2: Python Script (Cross-platform)
```bash
cd advanced
python3 install.py
```

## Prerequisites

- **Python 3.8+** (Required)
- **Git** (For cloning the repository)
- **Internet connection** (For downloading packages)

### System-specific Requirements

#### Linux/Ubuntu
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python
```

#### Windows
- Download Python from [python.org](https://python.org)
- Ensure Python is added to PATH during installation

## Installation Options

### 1. Automatic Installation (Recommended)

The installation scripts will:
- ✅ Check Python version compatibility
- ✅ Create virtual environment
- ✅ Install all required dependencies
- ✅ Initialize database schema
- ✅ Run tests to verify installation
- ✅ Create necessary directories
- ✅ Set up configuration files

#### Bash Script Features
```bash
./install.sh
```

#### Python Script Features
```bash
# Basic installation
python3 install.py

# Skip tests (faster installation)
python3 install.py --skip-tests

# Force database reinitialization
python3 install.py --force-reinit

# Specify custom project root
python3 install.py --project-root /path/to/project
```

### 2. Manual Installation

If you prefer manual installation or encounter issues:

#### Step 1: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

#### Step 2: Upgrade pip
```bash
pip install --upgrade pip
```

#### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Configure Environment
```bash
cp .env.sample .env
# Edit .env with your configuration
```

#### Step 5: Initialize Database
```bash
python main.py --init
```

## Configuration

### Environment Variables

Copy `.env.sample` to `.env` and configure:

```bash
# PerplexityAI MCP Configuration
PERPLEXITY_MCP_URL=https://mcp.composio.dev/composio/server/YOUR-SERVER-ID
PERPLEXITY_MCP_SERVER_NAME=perplexityai

# Database Configuration
DB_PATH=./duckdb_advanced.db
DB_MEMORY_LIMIT=4GB
DB_THREADS=4
DB_CACHE_ENABLED=true

# Logging Configuration
LOG_LEVEL=INFO
```

### Required Configuration Steps

1. **PerplexityAI MCP Server**: Update `PERPLEXITY_MCP_URL` with your actual server URL
2. **API Keys**: Add any required API keys (if needed)
3. **Resource Limits**: Adjust memory and thread settings based on your system

## Verification

### 1. Run Tests
```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Run tests
pytest tests/ -v
```

### 2. Run Demo
```bash
python main.py
```

### 3. Check Database
```bash
python main.py --init
```

## Usage Examples

After installation, use the runner scripts for easy execution:

### Option 1: Bash Runner (Linux/macOS)
```bash
./run.sh --help              # Show help
./run.sh --init              # Initialize database
./run.sh --gather            # Gather data from PerplexityAI
./run.sh --response-file response.json  # Process response data
./run.sh                     # Run interactive demo
```

### Option 2: Python Runner (Cross-platform)
```bash
python3 run.py --help        # Show help
python3 run.py --init        # Initialize database
python3 run.py --gather      # Gather data from PerplexityAI
python3 run.py --response-file response.json  # Process response data
python3 run.py               # Run interactive demo
```

### Option 3: Manual Execution (Advanced)
```bash
# Activate virtual environment and run from parent directory
. venv/bin/activate
cd ..
python -m advanced.main --init
```

## Directory Structure

After installation, you'll have:

```
advanced/
├── install.sh              # Bash installation script
├── install.py              # Python installation script
├── requirements.txt         # Python dependencies
├── main.py                 # Main application entry point
├── config.py               # Configuration settings
├── .env                    # Environment variables (create from .env.sample)
├── .env.sample             # Environment template
├── venv/                   # Virtual environment (created during install)
├── logs/                   # Application and installation logs
├── exports/                # Data export directory
├── visualizations/         # Visualization output directory
├── data/                   # Data storage directory
├── backups/                # Database backup directory
├── db/                     # Database modules
├── models/                 # Data models and schemas
├── perplexity/             # PerplexityAI integration
├── analytics/              # Analytics modules
├── utils/                  # Utility functions
└── tests/                  # Test suite
```

## Troubleshooting

### Common Issues

#### 1. Python Version Error
```
Error: Python 3.8+ required, found 3.7.x
```
**Solution**: Upgrade Python to 3.8 or higher

#### 2. Permission Denied
```
Error: Permission denied when creating virtual environment
```
**Solution**: 
```bash
chmod +x install.sh
# or
chmod +x install.py
```

#### 3. Package Installation Fails
```
Error: Failed to install packages
```
**Solutions**:
- Ensure internet connection
- Update pip: `pip install --upgrade pip`
- Try installing packages individually
- Check for conflicting Python installations

#### 4. Database Initialization Fails
```
Error: Failed to initialize database
```
**Solutions**:
- Check file permissions in project directory
- Ensure virtual environment is activated
- Verify all dependencies are installed
- Check disk space availability

#### 5. Environment Configuration Issues
```
Warning: .env file needs configuration
```
**Solution**: Edit `.env` file with your actual configuration values

### Getting Help

1. **Check Logs**: Review `logs/install.log` for detailed error information
2. **Run Tests**: Execute `pytest tests/ -v` to identify specific issues
3. **Verify Environment**: Ensure all environment variables are properly set
4. **Check Dependencies**: Verify all required packages are installed

### Manual Cleanup

If you need to start fresh:

```bash
# Remove virtual environment
rm -rf venv/

# Remove database
rm -f duckdb_advanced.db

# Remove logs
rm -rf logs/

# Re-run installation
./install.sh
# or
python3 install.py
```

## Advanced Configuration

### Performance Tuning

Edit `.env` file:

```bash
# Increase memory limit for large datasets
DB_MEMORY_LIMIT=8GB

# Increase thread count for faster processing
DB_THREADS=8

# Enable advanced caching
DB_CACHE_ENABLED=true
DB_CACHE_SIZE=200
```

### Development Setup

For development work:

```bash
# Install development dependencies
pip install black flake8 mypy pytest-cov

# Install in development mode
pip install -e .
```

## Security Considerations

- **Never commit `.env` files** to version control
- **Use environment variables** for sensitive configuration
- **Regular updates**: Keep dependencies updated for security patches
- **File permissions**: Ensure proper file permissions on production systems

## Next Steps

After successful installation:

1. Configure your PerplexityAI MCP server details
2. Run the demo to verify functionality
3. Explore the analytics capabilities
4. Set up data gathering workflows
5. Review the documentation for advanced features

## Support

For issues or questions:
- Check the troubleshooting section above
- Review installation logs in `logs/install.log`
- Run tests to identify specific problems
- Ensure all prerequisites are met
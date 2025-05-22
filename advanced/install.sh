#!/bin/bash

# Advanced DuckDB Installation and Setup Script
# This script initializes the database and installs all requirements for the advanced DuckDB implementation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Log file
LOG_FILE="$PROJECT_ROOT/logs/install.log"

# Ensure logs directory exists
mkdir -p "$PROJECT_ROOT/logs"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    print_status "Checking Python version..."
    
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
        PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION detected"
            return 0
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

# Function to check if virtual environment exists
check_virtual_env() {
    if [ -d "$PROJECT_ROOT/venv" ]; then
        print_success "Virtual environment found"
        return 0
    else
        print_warning "Virtual environment not found"
        return 1
    fi
}

# Function to create virtual environment
create_virtual_env() {
    print_status "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/venv"
    print_success "Virtual environment created"
}

# Function to activate virtual environment
activate_virtual_env() {
    print_status "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
    print_success "Virtual environment activated"
}

# Function to upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    pip install --upgrade pip
    print_success "Pip upgraded"
}

# Function to install Python packages
install_python_packages() {
    print_status "Installing Python packages..."
    
    # Core packages based on code analysis
    PACKAGES=(
        "duckdb>=0.9.0"
        "pandas>=1.5.0"
        "numpy>=1.20.0"
        "scikit-learn>=1.0.0"
        "python-dotenv>=0.19.0"
        "pytest>=7.0.0"
        "pytest-cov>=4.0.0"
        "matplotlib>=3.5.0"
        "seaborn>=0.11.0"
        "plotly>=5.0.0"
        "requests>=2.28.0"
        "aiohttp>=3.8.0"
        "pydantic>=1.10.0"
        "typing-extensions>=4.0.0"
    )
    
    for package in "${PACKAGES[@]}"; do
        print_status "Installing $package..."
        pip install "$package" || {
            print_error "Failed to install $package"
            return 1
        }
    done
    
    print_success "All Python packages installed"
}

# Function to create requirements.txt
create_requirements_file() {
    print_status "Creating requirements.txt..."
    
    cat > "$PROJECT_ROOT/requirements.txt" << EOF
# Core dependencies for Advanced DuckDB Implementation
duckdb>=0.9.0
pandas>=1.5.0
numpy>=1.20.0
scikit-learn>=1.0.0
python-dotenv>=0.19.0

# Testing dependencies
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0

# Visualization dependencies
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0

# Web and API dependencies
requests>=2.28.0
aiohttp>=3.8.0

# Data validation and serialization
pydantic>=1.10.0
typing-extensions>=4.0.0

# Optional: Development tools
black>=22.0.0
flake8>=5.0.0
mypy>=0.991
EOF
    
    print_success "requirements.txt created"
}

# Function to check environment file
check_env_file() {
    print_status "Checking environment configuration..."
    
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        print_warning ".env file not found, creating from .env.sample..."
        cp "$PROJECT_ROOT/.env.sample" "$PROJECT_ROOT/.env"
        print_warning "Please edit .env file with your actual configuration values"
        return 1
    else
        print_success ".env file found"
        return 0
    fi
}

# Function to check database initialization
check_database() {
    print_status "Checking database status..."
    
    if [ -f "$PROJECT_ROOT/duckdb_advanced.db" ]; then
        print_warning "Database file already exists"
        read -p "Do you want to reinitialize the database? (y/N): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$PROJECT_ROOT/duckdb_advanced.db"
            print_status "Existing database removed"
            return 1
        else
            print_status "Keeping existing database"
            return 0
        fi
    else
        print_status "No existing database found"
        return 1
    fi
}

# Function to initialize database
initialize_database() {
    print_status "Initializing database..."
    
    cd "$PROJECT_ROOT"
    python3 -c "
import sys
sys.path.insert(0, '.')
from main import init_database
try:
    db = init_database()
    print('Database initialized successfully')
except Exception as e:
    print(f'Error initializing database: {e}')
    sys.exit(1)
" || {
        print_error "Failed to initialize database"
        return 1
    }
    
    print_success "Database initialized successfully"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    cd "$PROJECT_ROOT"
    if command_exists pytest; then
        pytest tests/ -v || {
            print_warning "Some tests failed, but installation can continue"
        }
        print_success "Tests completed"
    else
        print_warning "pytest not available, skipping tests"
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    DIRECTORIES=(
        "logs"
        "exports"
        "visualizations"
        "data"
        "backups"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        mkdir -p "$PROJECT_ROOT/$dir"
        print_status "Created directory: $dir"
    done
    
    print_success "All directories created"
}

# Function to set permissions
set_permissions() {
    print_status "Setting file permissions..."
    
    chmod +x "$PROJECT_ROOT/install.sh"
    chmod +x "$PROJECT_ROOT/main.py"
    chmod +x "$PROJECT_ROOT/init_duckdb.py"
    
    print_success "File permissions set"
}

# Function to display completion message
display_completion() {
    echo
    echo "=========================================="
    print_success "Installation completed successfully!"
    echo "=========================================="
    echo
    print_status "Next steps:"
    echo "1. Edit .env file with your configuration"
    echo "2. Run: python3 main.py --init (to initialize database)"
    echo "3. Run: python3 main.py --gather (to gather data)"
    echo "4. Run: python3 main.py (for demo)"
    echo
    print_status "Available commands:"
    echo "  python3 main.py --help          - Show help"
    echo "  python3 main.py --init          - Initialize database"
    echo "  python3 main.py --gather        - Gather data from PerplexityAI"
    echo "  python3 main.py                 - Run demo"
    echo "  pytest tests/                   - Run tests"
    echo
    print_status "Installation log saved to: $LOG_FILE"
}

# Main installation function
main() {
    echo "=========================================="
    echo "Advanced DuckDB Installation Script"
    echo "=========================================="
    echo
    
    # Log start time
    echo "Installation started at $(date)" > "$LOG_FILE"
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    if ! check_python_version; then
        print_error "Python 3.8+ is required"
        exit 1
    fi
    
    # Create directories
    create_directories
    
    # Setup virtual environment
    if ! check_virtual_env; then
        create_virtual_env
    fi
    
    activate_virtual_env
    upgrade_pip
    
    # Create requirements file if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
        create_requirements_file
    fi
    
    # Install packages
    install_python_packages
    
    # Check environment
    check_env_file
    ENV_STATUS=$?
    
    # Set permissions
    set_permissions
    
    # Check and initialize database
    if ! check_database; then
        initialize_database
    fi
    
    # Run tests
    run_tests
    
    # Display completion message
    display_completion
    
    # Log completion
    echo "Installation completed at $(date)" >> "$LOG_FILE"
    
    # Exit with warning if .env needs configuration
    if [ $ENV_STATUS -eq 1 ]; then
        print_warning "Please configure .env file before using the application"
        exit 2
    fi
}

# Run main function
main "$@"
#!/bin/bash

# Advanced DuckDB Runner Script
# This script activates the virtual environment and runs the application correctly

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Advanced DuckDB Implementation Runner${NC}"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Virtual environment not found. Please run install.sh or install.py first."
    exit 1
fi

# Activate virtual environment and run from parent directory
cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/venv/bin/activate"

echo -e "${GREEN}Virtual environment activated${NC}"
echo "Running: python -m advanced.main $@"
echo

# Run the application with all passed arguments
python -m advanced.main "$@"
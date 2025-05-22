#!/usr/bin/env python3
"""
Advanced DuckDB Runner Script (Python version)
This script activates the virtual environment and runs the application correctly
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def main():
    """Main runner function."""
    # Get script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Virtual environment directory
    venv_dir = script_dir / "venv"
    
    print("Advanced DuckDB Implementation Runner")
    print("=" * 38)
    
    # Check if virtual environment exists
    if not venv_dir.exists():
        print("Virtual environment not found. Please run install.sh or install.py first.")
        sys.exit(1)
    
    # Get Python executable from virtual environment
    if platform.system() == "Windows":
        python_exe = venv_dir / "Scripts" / "python.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
    
    if not python_exe.exists():
        print(f"Python executable not found at {python_exe}")
        sys.exit(1)
    
    print("Virtual environment activated")
    
    # Prepare command
    args = sys.argv[1:]  # Get all command line arguments except script name
    command = [str(python_exe), "-m", "advanced.main"] + args
    
    print(f"Running: {' '.join(command[1:])}")  # Don't show full python path
    print()
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Run the application
    try:
        result = subprocess.run(command, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
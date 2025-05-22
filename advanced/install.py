#!/usr/bin/env python3
"""
Advanced DuckDB Installation and Setup Script (Python version)
This script provides cross-platform installation and initialization for the advanced DuckDB implementation.
"""

import os
import sys
import subprocess
import shutil
import platform
import argparse
from pathlib import Path
from typing import List, Optional, Tuple


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class Installer:
    """Main installer class for the advanced DuckDB implementation."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the installer."""
        self.project_root = project_root or Path(__file__).parent
        self.log_file = self.project_root / "logs" / "install.log"
        self.requirements_file = self.project_root / "requirements.txt"
        self.venv_dir = self.project_root / "venv"
        self.env_file = self.project_root / ".env"
        self.env_sample = self.project_root / ".env.sample"
        
        # Ensure logs directory exists
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Clear log file
        with open(self.log_file, 'w') as f:
            f.write(f"Installation started at {self._get_timestamp()}\n")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _log_and_print(self, message: str, level: str = "INFO") -> None:
        """Log message to file and print to console."""
        color_map = {
            "INFO": Colors.BLUE,
            "SUCCESS": Colors.GREEN,
            "WARNING": Colors.YELLOW,
            "ERROR": Colors.RED
        }
        
        color = color_map.get(level, Colors.NC)
        formatted_message = f"{color}[{level}]{Colors.NC} {message}"
        
        print(formatted_message)
        
        with open(self.log_file, 'a') as f:
            f.write(f"[{self._get_timestamp()}] [{level}] {message}\n")
    
    def print_status(self, message: str) -> None:
        """Print status message."""
        self._log_and_print(message, "INFO")
    
    def print_success(self, message: str) -> None:
        """Print success message."""
        self._log_and_print(message, "SUCCESS")
    
    def print_warning(self, message: str) -> None:
        """Print warning message."""
        self._log_and_print(message, "WARNING")
    
    def print_error(self, message: str) -> None:
        """Print error message."""
        self._log_and_print(message, "ERROR")
    
    def command_exists(self, command: str) -> bool:
        """Check if a command exists in the system."""
        return shutil.which(command) is not None
    
    def run_command(self, command: List[str], cwd: Optional[Path] = None, check: bool = True) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                check=check
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return e.returncode, e.stdout, e.stderr
        except Exception as e:
            return 1, "", str(e)
    
    def check_python_version(self) -> bool:
        """Check if Python version is 3.8+."""
        self.print_status("Checking Python version...")
        
        if sys.version_info >= (3, 8):
            version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            self.print_success(f"Python {version} detected")
            return True
        else:
            version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            self.print_error(f"Python 3.8+ required, found {version}")
            return False
    
    def check_virtual_env(self) -> bool:
        """Check if virtual environment exists."""
        if self.venv_dir.exists():
            self.print_success("Virtual environment found")
            return True
        else:
            self.print_warning("Virtual environment not found")
            return False
    
    def create_virtual_env(self) -> bool:
        """Create virtual environment."""
        self.print_status("Creating virtual environment...")
        
        exit_code, stdout, stderr = self.run_command([
            sys.executable, "-m", "venv", str(self.venv_dir)
        ])
        
        if exit_code == 0:
            self.print_success("Virtual environment created")
            return True
        else:
            self.print_error(f"Failed to create virtual environment: {stderr}")
            return False
    
    def get_pip_executable(self) -> Path:
        """Get the pip executable path for the virtual environment."""
        if platform.system() == "Windows":
            return self.venv_dir / "Scripts" / "pip.exe"
        else:
            return self.venv_dir / "bin" / "pip"
    
    def get_python_executable(self) -> Path:
        """Get the Python executable path for the virtual environment."""
        if platform.system() == "Windows":
            return self.venv_dir / "Scripts" / "python.exe"
        else:
            return self.venv_dir / "bin" / "python"
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip in the virtual environment."""
        self.print_status("Upgrading pip...")
        
        pip_exe = self.get_pip_executable()
        exit_code, stdout, stderr = self.run_command([
            str(pip_exe), "install", "--upgrade", "pip"
        ])
        
        if exit_code == 0:
            self.print_success("Pip upgraded")
            return True
        else:
            self.print_error(f"Failed to upgrade pip: {stderr}")
            return False
    
    def create_requirements_file(self) -> None:
        """Create requirements.txt file."""
        self.print_status("Creating requirements.txt...")
        
        requirements = """# Core dependencies for Advanced DuckDB Implementation
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
"""
        
        with open(self.requirements_file, 'w') as f:
            f.write(requirements)
        
        self.print_success("requirements.txt created")
    
    def install_python_packages(self) -> bool:
        """Install Python packages from requirements.txt."""
        self.print_status("Installing Python packages...")
        
        pip_exe = self.get_pip_executable()
        
        if not self.requirements_file.exists():
            self.create_requirements_file()
        
        exit_code, stdout, stderr = self.run_command([
            str(pip_exe), "install", "-r", str(self.requirements_file)
        ])
        
        if exit_code == 0:
            self.print_success("All Python packages installed")
            return True
        else:
            self.print_error(f"Failed to install packages: {stderr}")
            return False
    
    def check_env_file(self) -> bool:
        """Check and create environment file."""
        self.print_status("Checking environment configuration...")
        
        if not self.env_file.exists():
            if self.env_sample.exists():
                self.print_warning(".env file not found, creating from .env.sample...")
                shutil.copy2(self.env_sample, self.env_file)
                self.print_warning("Please edit .env file with your actual configuration values")
                return False
            else:
                self.print_error(".env.sample file not found")
                return False
        else:
            self.print_success(".env file found")
            return True
    
    def check_database(self) -> bool:
        """Check database status."""
        self.print_status("Checking database status...")
        
        db_file = self.project_root / "duckdb_advanced.db"
        
        if db_file.exists():
            self.print_warning("Database file already exists")
            response = input("Do you want to reinitialize the database? (y/N): ")
            if response.lower() in ['y', 'yes']:
                db_file.unlink()
                self.print_status("Existing database removed")
                return False
            else:
                self.print_status("Keeping existing database")
                return True
        else:
            self.print_status("No existing database found")
            return False
    
    def initialize_database(self) -> bool:
        """Initialize the database."""
        self.print_status("Initializing database...")
        
        python_exe = self.get_python_executable()
        
        # Create initialization script
        init_script = """
import sys
sys.path.insert(0, '.')
try:
    from main import init_database
    db = init_database()
    print('Database initialized successfully')
except Exception as e:
    print(f'Error initializing database: {e}')
    sys.exit(1)
"""
        
        exit_code, stdout, stderr = self.run_command([
            str(python_exe), "-c", init_script
        ])
        
        if exit_code == 0:
            self.print_success("Database initialized successfully")
            return True
        else:
            self.print_error(f"Failed to initialize database: {stderr}")
            return False
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        self.print_status("Creating necessary directories...")
        
        directories = [
            "logs",
            "exports", 
            "visualizations",
            "data",
            "backups"
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            self.print_status(f"Created directory: {dir_name}")
        
        self.print_success("All directories created")
    
    def run_tests(self) -> None:
        """Run tests."""
        self.print_status("Running tests...")
        
        python_exe = self.get_python_executable()
        tests_dir = self.project_root / "tests"
        
        if tests_dir.exists():
            exit_code, stdout, stderr = self.run_command([
                str(python_exe), "-m", "pytest", str(tests_dir), "-v"
            ], check=False)
            
            if exit_code == 0:
                self.print_success("All tests passed")
            else:
                self.print_warning("Some tests failed, but installation can continue")
        else:
            self.print_warning("Tests directory not found, skipping tests")
    
    def display_completion(self, env_configured: bool) -> None:
        """Display completion message."""
        print("\n" + "=" * 50)
        self.print_success("Installation completed successfully!")
        print("=" * 50 + "\n")
        
        self.print_status("Next steps:")
        if not env_configured:
            print("1. Edit .env file with your configuration")
            print("2. Run: python main.py --init (to initialize database)")
        else:
            print("1. Run: python main.py --init (to initialize database)")
        print("2. Run: python main.py --gather (to gather data)")
        print("3. Run: python main.py (for demo)")
        print()
        
        self.print_status("Available commands:")
        print("  python main.py --help          - Show help")
        print("  python main.py --init          - Initialize database")
        print("  python main.py --gather        - Gather data from PerplexityAI")
        print("  python main.py                 - Run demo")
        print("  pytest tests/                  - Run tests")
        print()
        
        self.print_status(f"Installation log saved to: {self.log_file}")
    
    def install(self, skip_tests: bool = False, force_reinit: bool = False) -> bool:
        """Run the complete installation process."""
        print("=" * 50)
        print("Advanced DuckDB Installation Script")
        print("=" * 50 + "\n")
        
        # Check prerequisites
        self.print_status("Checking prerequisites...")
        
        if not self.check_python_version():
            self.print_error("Python 3.8+ is required")
            return False
        
        # Create directories
        self.create_directories()
        
        # Setup virtual environment
        if not self.check_virtual_env():
            if not self.create_virtual_env():
                return False
        
        if not self.upgrade_pip():
            return False
        
        # Install packages
        if not self.install_python_packages():
            return False
        
        # Check environment
        env_configured = self.check_env_file()
        
        # Check and initialize database
        if force_reinit or not self.check_database():
            if not self.initialize_database():
                return False
        
        # Run tests
        if not skip_tests:
            self.run_tests()
        
        # Display completion message
        self.display_completion(env_configured)
        
        # Log completion
        with open(self.log_file, 'a') as f:
            f.write(f"Installation completed at {self._get_timestamp()}\n")
        
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Advanced DuckDB Installation Script")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--force-reinit", action="store_true", help="Force database reinitialization")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    
    args = parser.parse_args()
    
    installer = Installer(args.project_root)
    
    try:
        success = installer.install(
            skip_tests=args.skip_tests,
            force_reinit=args.force_reinit
        )
        
        if success:
            installer.print_success("Installation completed successfully!")
            sys.exit(0)
        else:
            installer.print_error("Installation failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        installer.print_warning("Installation interrupted by user")
        sys.exit(130)
    except Exception as e:
        installer.print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
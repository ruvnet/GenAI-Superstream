"""
Test runner for the GenAI-Superstream project.

This script discovers and runs all tests in the tests directory,
or specific tests as requested by command-line arguments.
"""

import unittest
import sys
import os
import argparse
import importlib
import logging

def setup_logging(level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level to use
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("test_runner")

def run_specific_test(test_name, logger):
    """
    Run a specific test case or test module.
    
    Args:
        test_name: Name of the test to run (module.TestCase.test_method)
        logger: Logger instance
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add parent directory to path to allow importing modules
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    parts = test_name.split('.')
    
    if len(parts) == 1:
        # Run an entire test module
        module_name = f"tests.{parts[0]}"
        try:
            logger.info(f"Importing test module: {module_name}")
            module = importlib.import_module(module_name)
            
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            test_runner = unittest.TextTestRunner(verbosity=2)
            result = test_runner.run(suite)
            return result.wasSuccessful()
        except ImportError as e:
            logger.error(f"Failed to import test module {module_name}: {e}")
            return False
    
    elif len(parts) == 2:
        # Run an entire test case
        module_name = f"tests.{parts[0]}"
        case_name = parts[1]
        
        try:
            logger.info(f"Importing test module: {module_name}")
            module = importlib.import_module(module_name)
            
            # Get the test case class
            case_class = getattr(module, case_name)
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(case_class)
            test_runner = unittest.TextTestRunner(verbosity=2)
            result = test_runner.run(suite)
            return result.wasSuccessful()
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to run test case {test_name}: {e}")
            return False
    
    elif len(parts) == 3:
        # Run a specific test method
        module_name = f"tests.{parts[0]}"
        case_name = parts[1]
        method_name = parts[2]
        
        try:
            logger.info(f"Importing test module: {module_name}")
            module = importlib.import_module(module_name)
            
            # Get the test case class
            case_class = getattr(module, case_name)
            suite = unittest.TestSuite()
            suite.addTest(case_class(method_name))
            test_runner = unittest.TextTestRunner(verbosity=2)
            result = test_runner.run(suite)
            return result.wasSuccessful()
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to run test method {test_name}: {e}")
            return False
    
    else:
        logger.error(f"Invalid test name format: {test_name}")
        return False

def run_tests(pattern="test_*.py", mcp_only=False, verbose=False):
    """
    Discover and run all tests matching the pattern.
    
    Args:
        pattern: Pattern to match test files
        mcp_only: If True, only run MCP-related tests
        verbose: If True, enable verbose output
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add parent directory to path to allow importing modules
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Discover and run tests
    test_loader = unittest.TestLoader()
    
    if mcp_only:
        # Only run MCP-related tests
        test_pattern = "test_mcp*.py"
        print(f"Running only MCP-related tests matching pattern: {test_pattern}")
        test_suite = test_loader.discover(script_dir, pattern=test_pattern)
    else:
        test_suite = test_loader.discover(script_dir, pattern=pattern)
    
    # Set verbosity level
    verbosity = 2 if verbose else 1
    
    test_runner = unittest.TextTestRunner(verbosity=verbosity)
    result = test_runner.run(test_suite)
    
    # Return True if successful, False otherwise
    return result.wasSuccessful()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run tests for GenAI-Superstream')
    parser.add_argument('--test', type=str, help='Specific test to run (module.TestCase.test_method)')
    parser.add_argument('--pattern', type=str, default='test_*.py', help='Pattern for test discovery')
    parser.add_argument('--mcp-only', action='store_true', help='Only run MCP-related tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level)
    
    if args.test:
        # Run a specific test
        success = run_specific_test(args.test, logger)
    else:
        # Run all tests matching pattern
        success = run_tests(args.pattern, args.mcp_only, args.verbose)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
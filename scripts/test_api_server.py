#!/usr/bin/env python3
"""
Test script for the standalone API server.

This script:
1. Starts the API server in a separate process
2. Tests various API endpoints
3. Reports detailed results
"""

import os
import sys
import time
import json
import logging
import requests
import subprocess
import threading
import signal
from contextlib import contextmanager

# Add project root to path so imports work properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("api_test")

@contextmanager
def run_server_in_subprocess(port=8000):
    """Start the API server in a separate process with proper cleanup."""
    logger.info(f"Starting API server on port {port}...")
    
    # Start server in subprocess
    server_process = subprocess.Popen(
        [sys.executable, "-m", "src.api_server", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    try:
        # Wait for server to start
        time.sleep(3)  # Give the server time to initialize
        
        logger.info("API server started")
        yield
        
    finally:
        # Terminate server
        logger.info("Terminating API server...")
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        
        # Get server output
        stdout, stderr = server_process.communicate(timeout=5)
        server_process.wait()
        
        logger.info("Server output:")
        logger.info(stdout.decode('utf-8'))
        
        if stderr:
            logger.info("Server error output:")
            logger.info(stderr.decode('utf-8'))
        
        logger.info("API server stopped")

def test_api_endpoints(port=8000):
    """Test all API endpoints with various input formats."""
    test_features = [5.1, 3.5, 1.4, 0.2]  # Example Iris setosa features
    
    # Test different endpoints and input formats
    test_cases = [
        {
            "name": "Direct features endpoint",
            "url": f"http://localhost:{port}/api/predict_species",
            "method": "POST",
            "payload": {"features": test_features}
        },
        {
            "name": "Gradio format - string input",
            "url": f"http://localhost:{port}/api/predict",
            "method": "POST",
            "payload": {"data": ["5.1, 3.5, 1.4, 0.2"]}
        },
        {
            "name": "Gradio format - array input",
            "url": f"http://localhost:{port}/api/predict",
            "method": "POST",
            "payload": {"data": [test_features]}
        },
        {
            "name": "MCP tool endpoint - direct path",
            "url": f"http://localhost:{port}/api/mcp/tools/predict_species",
            "method": "POST",
            "payload": {"features": test_features}
        },
        {
            "name": "MCP standard path",
            "url": f"http://localhost:{port}/mcp/api/v1/tools/predict_species",
            "method": "POST",
            "payload": {"features": test_features}
        },
        {
            "name": "MCP call format",
            "url": f"http://localhost:{port}/mcp/call",
            "method": "POST",
            "payload": {
                "tool": "predict_species",
                "input": {"features": test_features}
            }
        },
        {
            "name": "Health check",
            "url": f"http://localhost:{port}/health",
            "method": "GET",
            "payload": None
        }
    ]
    
    results = []
    
    # Run all test cases
    for test_case in test_cases:
        logger.info(f"Testing {test_case['name']} at {test_case['url']}")
        
        try:
            if test_case["method"] == "GET":
                response = requests.get(test_case["url"], timeout=5)
            else:
                response = requests.post(
                    test_case["url"],
                    json=test_case["payload"],
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
            
            # Log response
            status = response.status_code
            content = response.text[:200]  # Truncate for readability
            
            success = status == 200
            result = {
                "test_case": test_case["name"],
                "url": test_case["url"],
                "success": success,
                "status": status,
                "content": content
            }
            
            if success:
                logger.info(f"✅ Success! Status {status}: {content}")
            else:
                logger.warning(f"❌ Failed! Status {status}: {content}")
                
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error testing {test_case['url']}: {e}")
            results.append({
                "test_case": test_case["name"],
                "url": test_case["url"],
                "success": False,
                "error": str(e)
            })
    
    # Summarize results
    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    
    logger.info(f"\nResults Summary: {success_count} / {total_count} tests passed")
    
    for result in results:
        status = "✅ Passed" if result["success"] else "❌ Failed"
        logger.info(f"{status}: {result['test_case']}")
    
    return success_count, total_count, results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the API server')
    parser.add_argument('--port', type=int, default=8000, help='Port to use for the API server')
    parser.add_argument('--no-start', action='store_true', help='Don\'t start the server (test external server)')
    
    args = parser.parse_args()
    
    if args.no_start:
        # Test an already running server
        logger.info(f"Testing against server on port {args.port} (not starting new server)")
        success_count, total_count, results = test_api_endpoints(args.port)
        
        # Exit with error if any tests failed
        if success_count < total_count:
            sys.exit(1)
    else:
        # Start the server and test it
        try:
            with run_server_in_subprocess(args.port):
                # Give the server a bit more time to start
                time.sleep(2)
                
                # Run the tests
                success_count, total_count, results = test_api_endpoints(args.port)
                
                # Exit with error if any tests failed
                if success_count < total_count:
                    sys.exit(1)
                    
        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()
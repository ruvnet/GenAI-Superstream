#!/usr/bin/env python3
"""
Direct test script to verify MCP server functionality.

This script:
1. Starts a server directly without using the MCP client
2. Sends HTTP requests to various API endpoints 
3. Reports detailed results
"""

import os
import sys
import json
import time
import logging
import requests
import threading
import signal
from contextlib import contextmanager

# Add project root to path so imports work properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.server import predict_species, create_app, create_mcp_server
from src.model import Predictor, ModelTrainer
from src.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("direct_test")

@contextmanager
def run_server_in_thread(port=7995):
    """Start a server in a separate thread with proper cleanup."""
    server_ready = threading.Event()
    server_error = threading.Event()
    server_error_message = []
    
    def run_server():
        try:
            # Initialize the predictor
            data_loader = DataLoader()
            _, _, _, target_names = data_loader.load_iris_dataset()
            
            trainer = ModelTrainer()
            X_train, X_test, y_train, y_test = data_loader.split_data()
            model = trainer.train(X_train, y_train)
            
            predictor = Predictor(model, target_names)
            
            # Ensure the predictor is attached
            predict_species._predictor = predictor
            
            # Create app - directly using create_app
            logger.info("Creating server for direct testing...")
            app = create_app()
            
            # Launch server with explicit parameters
            logger.info(f"Launching server on port {port}...")
            server = app.launch(
                server_name="127.0.0.1",
                server_port=port,
                mcp_server=True,  # Enable MCP server
                show_api=True,     # Expose the API endpoints
                share=False,
                prevent_thread_lock=True  # Don't block this thread
            )
            
            # Signal that server is ready
            server_ready.set()
            logger.info(f"Server started at http://127.0.0.1:{port}")
            
            # Keep thread alive
            while not server_error.is_set():
                time.sleep(0.1)
                
            logger.info("Shutting down server...")
            # Handle different return types from app.launch()
            if hasattr(server, 'close'):
                server.close()
            else:
                logger.warning("No clean way to close the server found")
                
        except Exception as e:
            logger.error(f"Server error: {e}")
            import traceback
            traceback.print_exc()
            server_error_message.append(str(e))
            server_error.set()
            raise
    
    # Start server in a thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to be ready or error
    for _ in range(30):  # 15 seconds timeout
        if server_ready.is_set() or server_error.is_set():
            break
        time.sleep(0.5)
    
    if server_error.is_set():
        raise RuntimeError(f"Server failed to start: {server_error_message}")
    
    if not server_ready.is_set():
        raise TimeoutError("Server did not start in time")
    
    try:
        # Wait a bit more to ensure server is fully ready
        time.sleep(2)
        yield
    finally:
        # Signal to shut down the server
        server_error.set()
        server_thread.join(timeout=5)
        logger.info("Server stopped")

def test_direct_api(port):
    """Test direct API endpoints with explicit HTTP requests."""
    test_features = [5.1, 3.5, 1.4, 0.2]  # Example Iris setosa features
    endpoints = [
        # Standard API endpoints
        {"url": f"http://127.0.0.1:{port}/api/predict", "method": "POST", 
         "payload": {"data": ["5.1, 3.5, 1.4, 0.2"]}},
        
        {"url": f"http://127.0.0.1:{port}/api/predict", "method": "POST", 
         "payload": {"data": [[5.1, 3.5, 1.4, 0.2]]}},
        
        # Direct feature endpoints
        {"url": f"http://127.0.0.1:{port}/api/predict_species", "method": "POST", 
         "payload": {"features": test_features}},
         
        # MCP endpoints
        {"url": f"http://127.0.0.1:{port}/gradio_api/mcp/api/v1/tools/predict_species", "method": "POST", 
         "payload": {"features": test_features}},
         
        {"url": f"http://127.0.0.1:{port}/mcp/api/v1/tools/predict_species", "method": "POST", 
         "payload": {"features": test_features}},
         
        {"url": f"http://127.0.0.1:{port}/api/mcp/tools/predict_species", "method": "POST", 
         "payload": {"features": test_features}}
    ]
    
    success = False
    success_details = None
    
    for endpoint in endpoints:
        try:
            logger.info(f"Testing endpoint: {endpoint['url']}")
            
            response = requests.request(
                method=endpoint["method"],
                url=endpoint["url"],
                json=endpoint["payload"],
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            status = response.status_code
            text = response.text[:300]  # Truncate for logging
            
            if status == 200:
                result = response.json()
                logger.info(f"✅ Success! {endpoint['url']} returned: {result}")
                success = True
                success_details = {
                    "endpoint": endpoint["url"],
                    "payload": endpoint["payload"],
                    "result": result
                }
                break
            else:
                logger.warning(f"Request failed with status {status}: {text}")
                
        except Exception as e:
            logger.error(f"Error testing {endpoint['url']}: {e}")
    
    return success, success_details

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Server API directly')
    parser.add_argument('--port', type=int, default=7995, help='Port to use for testing')
    args = parser.parse_args()
    
    try:
        with run_server_in_thread(args.port):
            # Give server time to initialize
            time.sleep(2)
            
            # Test API endpoints
            logger.info("Testing direct API calls...")
            success, details = test_direct_api(args.port)
            
            if success:
                logger.info(f"✅ Test passed! Successfully called the API")
                logger.info(f"Endpoint: {details['endpoint']}")
                logger.info(f"Payload: {details['payload']}")
                logger.info(f"Result: {details['result']}")
            else:
                logger.error("❌ All API calls failed!")
                
                # Fallback to trying a custom endpoint - direct access via predict_species function
                logger.info("Attempting manual prediction with built-in function")
                try:
                    # Import the prediction function
                    from src.server import predict_species
                    from src.data import DataLoader
                    from src.model import ModelTrainer, Predictor
                    
                    # Create a predictor if needed
                    if not hasattr(predict_species, "_predictor") or predict_species._predictor is None:
                        logger.info("Setting up predictor for direct call")
                        data_loader = DataLoader()
                        _, _, _, target_names = data_loader.load_iris_dataset()
                        trainer = ModelTrainer()
                        X_train, X_test, y_train, y_test = data_loader.split_data()
                        model = trainer.train(X_train, y_train)
                        predictor = Predictor(model, target_names)
                        predict_species._predictor = predictor
                    
                    # Try a direct call
                    logger.info("Making direct function call")
                    result = predict_species([5.1, 3.5, 1.4, 0.2])
                    logger.info(f"✅ Direct function call successful: {result}")
                except Exception as e:
                    logger.error(f"Error with direct function call: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
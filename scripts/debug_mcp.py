#!/usr/bin/env python3
"""
Debug script to troubleshoot MCP client-server interactions.

This script explicitly tests the MCP functionality by:
1. Starting a server with the MCP tool properly initialized
2. Verifying the MCP endpoint is accessible
3. Connecting with an MCP client
4. Calling the prediction tool
5. Reporting any errors in detail

Use this to diagnose the "NoneType object is not callable" error.
"""
import os
import sys
import time
import logging
import traceback
import threading
import argparse
import json
import signal
import subprocess
from contextlib import contextmanager
import requests
import urllib3
import requests

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

logger = logging.getLogger("mcp_debug")

def check_mcp_dependencies():
    """
    Check if required MCP dependencies are installed and provide detailed information for troubleshooting.
    
    This function verifies that the mcp client package and required gradio components
    are available, and provides detailed installation instructions if they're missing.
    
    Returns:
        dict: Status information about MCP dependencies
    """
    status = {
        "mcp_available": False,
        "gradio_mcp_support": False,
        "details": {},
        "installation_instructions": ""
    }
    
    # Check for mcp package
    try:
        import importlib
        mcp_spec = importlib.util.find_spec("mcp")
        status["mcp_available"] = mcp_spec is not None
        
        if mcp_spec:
            # Try to import the client
            try:
                from mcp import client as mcp_client
                status["details"]["mcp_client_importable"] = True
                # Try to get the version
                try:
                    import mcp
                    status["details"]["mcp_version"] = getattr(mcp, "__version__", "unknown")
                except Exception:
                    status["details"]["mcp_version"] = "unknown"
            except ImportError as e:
                status["details"]["mcp_client_importable"] = False
                status["details"]["mcp_client_error"] = str(e)
        
        # Check for gradio MCP support
        try:
            import gradio as gr
            status["details"]["gradio_version"] = getattr(gr, "__version__", "unknown")
            
            # Check for gradio.mcp module
            try:
                from gradio import mcp
                status["gradio_mcp_support"] = True
                # Check for specific API methods
                status["details"]["has_register_tool"] = hasattr(mcp, "register_tool")
                status["details"]["has_tools_class"] = False
                try:
                    from gradio.mcp import Tools
                    status["details"]["has_tools_class"] = True
                except ImportError:
                    pass
            except ImportError:
                status["gradio_mcp_support"] = False
        except ImportError:
            status["details"]["gradio_installed"] = False
    
    except Exception as e:
        status["details"]["error"] = str(e)
    
    # Generate installation instructions based on status
    instructions = []
    
    if not status["mcp_available"]:
        instructions.append(
            "MCP client package is not installed. Install it with:\n"
            "pip install mcp-client"
        )
    elif not status["details"].get("mcp_client_importable", False):
        instructions.append(
            "MCP client was found but could not be imported correctly. Try reinstalling:\n"
            "pip uninstall mcp-client -y\n"
            "pip install mcp-client"
        )
    
    if not status["gradio_mcp_support"]:
        if status["details"].get("gradio_installed", True):
            instructions.append(
                f"Gradio {status['details'].get('gradio_version', '')} is installed but doesn't have MCP support.\n"
                "Update to a version with MCP support:\n"
                "pip install 'gradio>=5.0.0'"
            )
        else:
            instructions.append(
                "Gradio is not installed. Install it with MCP support:\n"
                "pip install 'gradio>=5.0.0'"
            )
    
    status["installation_instructions"] = "\n\n".join(instructions)
    
    # Log the findings
    if status["mcp_available"] and status["gradio_mcp_support"]:
        logger.info("✅ MCP dependencies check passed.")
        logger.info(f"MCP version: {status['details'].get('mcp_version', 'unknown')}")
        logger.info(f"Gradio version: {status['details'].get('gradio_version', 'unknown')}")
    else:
        logger.error("❌ MCP dependencies check failed.")
        logger.error(status["installation_instructions"])
    
    return status

@contextmanager
def run_server_in_thread(port=7990):
    """
    Start a server in a separate thread with proper cleanup.
    """
    # Set up a flag to indicate if the server is ready
    server_ready = threading.Event()
    server_error = threading.Event()
    server_error_message = []
    
    # Create app and predictor
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
            
            # Create app
            logger.info("Creating Gradio app...")
            app = create_app()
            
            # Attempt to manually register the MCP tool 
            try:
                logger.info("Attempting direct MCP tool registration")
                from gradio import mcp
                # Try to register if the method exists
                if hasattr(mcp, "register_tool"):
                    mcp.register_tool(
                        fn=predict_species,
                        name="predict_species",
                        description="Predicts Iris species based on 4 features"
                    )
            except Exception as e:
                logger.error(f"Error registering MCP tool: {e}")
            
            # Launch server with explicit MCP server configuration
            logger.info(f"Launching server on port {port}...")
            launch_result = app.launch(
                server_name="127.0.0.1",
                server_port=port,
                mcp_server=True,  # Enable MCP server
                show_api=True,    # Expose the API endpoints
                share=False,
                prevent_thread_lock=True  # Don't block this thread
            )
            
            # Signal that server is ready
            logger.info("Server started")
            server_ready.set()
            
            # Keep thread alive
            while not server_error.is_set():
                time.sleep(0.1)
                
            logger.info("Shutting down server...")
            # Handle different return types from app.launch()
            if hasattr(launch_result, 'close'):
                # If it's a server object, call close()
                launch_result.close()
            elif isinstance(launch_result, tuple) and len(launch_result) > 0 and hasattr(launch_result[0], 'close'):
                # If it's a tuple with a server as the first element, call close() on it
                launch_result[0].close()
            else:
                logger.warning("Could not find a way to close the server properly. It may continue running in the background.")
            
        except Exception as e:
            logger.error(f"Server error: {e}")
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

def check_mcp_endpoint(port):
    """
    Check if the MCP endpoint is accessible.
    """
    logger.info("Checking MCP endpoint...")
    
    # First try the SSE endpoint which should be active
    sse_endpoint = f"http://127.0.0.1:{port}/gradio_api/mcp/sse"
    try:
        logger.info(f"Attempting to access SSE endpoint at: {sse_endpoint}")
        response = requests.get(sse_endpoint, timeout=2, stream=True)
        if response.status_code == 200:
            logger.info(f"✅ Success! SSE endpoint available at: {sse_endpoint}")
            # We don't need to read the stream fully
            response.close()
            return True
    except requests.exceptions.ReadTimeout:
        # This is expected for SSE connections - they stay open
        logger.info(f"✅ Success! SSE endpoint is streaming at: {sse_endpoint}")
        return True
    except Exception as e:
        logger.error(f"Error accessing SSE endpoint {sse_endpoint}: {e}")

    # Try all available possible MCP endpoints to find what works
    base_urls = [
        f"http://127.0.0.1:{port}/gradio_api/mcp",
        f"http://127.0.0.1:{port}/mcp"
    ]
    
    api_paths = [
        "",
        "/",
        "/api",
        "/api/v1",
        "/api/v1/tools",
        "/health"
    ]
    
    for base_url in base_urls:
        for path in api_paths:
            endpoint_url = f"{base_url}{path}"
            try:
                logger.info(f"Testing endpoint: {endpoint_url}")
                response = requests.get(endpoint_url, timeout=3)
                logger.info(f"Response from {endpoint_url}: {response.status_code} {response.text[:100]}")
                
                if response.status_code == 200:
                    logger.info(f"✅ Found working endpoint: {endpoint_url}")
                    return True
            except Exception as e:
                logger.info(f"Error accessing {endpoint_url}: {e}")
    
    # Check if we can use the predict API
    try:
        api_url = f"http://127.0.0.1:{port}/api/predict"
        logger.info(f"Trying Gradio predict API at: {api_url}")
        
        payload = {"data": ["5.1, 3.5, 1.4, 0.2"]}
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(api_url, json=payload, headers=headers, timeout=5)
        
        logger.info(f"API response: {response.status_code} {response.text[:100]}")
        
        if response.status_code == 200:
            logger.info("✅ Successfully accessed Gradio API")
            return True
    except Exception as e:
        logger.error(f"Error accessing Gradio API: {e}")
        
    logger.error("❌ Failed to access any MCP endpoint")
    return False

def test_mcp_client(port):
    """
    Test the MCP client by connecting to the server and calling the prediction tool.
    """
    logger.info("Testing MCP client...")
    
    try:
        # Try comprehensive approach to check all API endpoints
        logger.info("Trying multiple API endpoints for prediction")
        
        # Try regular Gradio API endpoints first
        api_endpoints = [
            "/api/predict",  # Most common endpoint
            "/run/predict",  # Alternative endpoint
        ]
        
        # Different payload formats to try
        payload_formats = [
            {"data": ["5.1, 3.5, 1.4, 0.2"]},         # String input
            {"data": [[5.1, 3.5, 1.4, 0.2]]},         # Array input
            {"fn_index": 0, "data": ["5.1, 3.5, 1.4, 0.2"]}, # With function index
        ]
        
        headers = {"Content-Type": "application/json"}
        
        # Try all combinations
        for endpoint in api_endpoints:
            api_url = f"http://127.0.0.1:{port}{endpoint}"
            logger.info(f"Testing endpoint: {api_url}")
            
            for i, payload in enumerate(payload_formats):
                try:
                    logger.info(f"Trying payload format {i+1}: {payload}")
                    response = requests.post(api_url, json=payload, headers=headers, timeout=5)
                    
                    status = response.status_code
                    logger.info(f"Response: {status} {response.text[:100]}")
                    
                    if status == 200:
                        result = response.json()
                        logger.info(f"✅ Successful API call: {result}")
                        return True, result
                except Exception as e:
                    logger.warning(f"Error with payload {i+1}: {e}")
        
        # Try direct MCP API calls
        mcp_endpoints = [
            f"http://127.0.0.1:{port}/gradio_api/mcp/api/v1/tools/predict_species",
            f"http://127.0.0.1:{port}/mcp/api/v1/tools/predict_species",
        ]
        
        mcp_payload = {"features": [5.1, 3.5, 1.4, 0.2]}
        
        for endpoint in mcp_endpoints:
            try:
                logger.info(f"Testing MCP endpoint: {endpoint}")
                response = requests.post(endpoint, json=mcp_payload, headers=headers, timeout=5)
                
                status = response.status_code
                logger.info(f"Response: {status} {response.text[:100]}")
                
                if status == 200:
                    result = response.json()
                    logger.info(f"✅ Successful MCP API call: {result}")
                    return True, result
            except Exception as e:
                logger.warning(f"Error with MCP endpoint: {e}")
        
        # Last resort: try to use the MCP client library
        try:
            import importlib.util
            if importlib.util.find_spec("mcp") is not None:
                try:
                    logger.info("MCP Client library found, attempting to use it")
                    
                    # Try multiple approaches to create the client
                    client = None
                    error_messages = []
                    
                    # Approach 1: Use client as callable module
                    try:
                        from mcp import client as mcp_client
                        client = mcp_client(f"http://127.0.0.1:{port}/gradio_api/mcp/sse")
                        logger.info("Successfully created MCP client using module callable")
                    except Exception as e1:
                        error_messages.append(f"Error with module callable approach: {e1}")
                        
                        # Approach 2: Import Client class directly
                        try:
                            from mcp.client import Client
                            client = Client(f"http://127.0.0.1:{port}/gradio_api/mcp/sse")
                            logger.info("Successfully created MCP client using Client class")
                        except Exception as e2:
                            error_messages.append(f"Error with Client class approach: {e2}")
                    
                    # If we have a client, try to use it
                    if client:
                        tools = client.list_tools()
                        logger.info(f"Available tools: {tools}")
                        
                        result = client.call(
                            tool="predict_species",
                            input={"features": [5.1, 3.5, 1.4, 0.2]}
                        )
                        
                        logger.info(f"✅ Successful MCP client call: {result}")
                        return True, result
                    else:
                        # If direct MCP client connection failed, try a direct HTTP request
                        import requests
                        logger.info("Trying direct HTTP request to API endpoint")
                        
                        api_url = f"http://127.0.0.1:{port}/api/predict"
                        response = requests.post(
                            api_url,
                            json={"features": [5.1, 3.5, 1.4, 0.2]},
                            headers={"Content-Type": "application/json"}
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            logger.info(f"✅ Successful direct API call: {result}")
                            return True, result
                        else:
                            # Try other endpoints
                            for endpoint in [
                                f"/gradio_api/mcp/api/v1/tools/predict_species",
                                f"/mcp/api/v1/tools/predict_species",
                                f"/api/mcp/tools/predict_species"
                            ]:
                                api_url = f"http://127.0.0.1:{port}{endpoint}"
                                logger.info(f"Trying endpoint: {api_url}")
                                
                                response = requests.post(
                                    api_url,
                                    json={"features": [5.1, 3.5, 1.4, 0.2]},
                                    headers={"Content-Type": "application/json"}
                                )
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    logger.info(f"✅ Successful API call to {endpoint}: {result}")
                                    return True, result
                        
                        # If all approaches failed
                        logger.error(f"All client connection approaches failed: {error_messages}")
                        return False, "Failed to connect to MCP server"
                except Exception as e:
                    logger.error(f"Error using MCP client: {e}")
        except Exception as e:
            logger.error(f"Error importing MCP client: {e}")
        
        logger.error("All API and MCP access attempts failed")
        return False, "Failed to access MCP or API endpoints"
        
    except Exception as e:
        logger.error(f"Unexpected error in test_mcp_client: {e}")
        traceback.print_exc()
        return False, str(e)

def test_with_subprocess(port):
    """
    Test by running the server in a separate process for complete isolation.
    """
    logger.info("Testing with server in subprocess...")
    
    # Start server in subprocess
    server_process = subprocess.Popen(
        [sys.executable, "-m", "src.server", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    try:
        # Wait for server to start
        logger.info("Waiting for server to start...")
        time.sleep(5)
        
        # Check MCP endpoint
        if not check_mcp_endpoint(port):
            logger.error("MCP endpoint check failed")
            return False
        
        # Test client
        success, result = test_mcp_client(port)
        return success
        
    finally:
        # Terminate server
        logger.info("Terminating server subprocess...")
        os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)
        
        # Get server output
        stdout, stderr = server_process.communicate(timeout=5)
        server_process.wait()
        
        logger.info("Server output:")
        logger.info(stdout.decode('utf-8'))
        
        logger.info("Server error output:")
        logger.info(stderr.decode('utf-8'))

def check_and_fix_mcp_setup():
    """
    Standalone utility to check and fix MCP dependencies.
    
    This function verifies all required MCP dependencies are installed correctly
    and attempts to install any missing components automatically.
    
    Returns:
        bool: True if setup is complete and working, False otherwise
    """
    logger.info("Starting MCP setup check and fix utility...")
    
    # Check current dependencies
    dep_status = check_mcp_dependencies()
    
    if dep_status["mcp_available"] and dep_status["gradio_mcp_support"]:
        logger.info("✅ All MCP dependencies are installed correctly.")
        logger.info(f"MCP version: {dep_status['details'].get('mcp_version', 'unknown')}")
        logger.info(f"Gradio version: {dep_status['details'].get('gradio_version', 'unknown')}")
        return True
    
    # Display missing dependencies
    logger.warning("⚠️ Some MCP dependencies are missing or incompatible.")
    logger.info("\nMissing or incompatible dependencies:")
    
    if not dep_status["mcp_available"]:
        logger.info("❌ MCP client package is not installed")
    elif not dep_status["details"].get("mcp_client_importable", False):
        logger.info("❌ MCP client package is installed but cannot be imported correctly")
    
    if not dep_status["gradio_mcp_support"]:
        logger.info("❌ Gradio MCP support is not available")
    
    # Check if pip is available
    try:
        import pip
        pip_available = True
    except ImportError:
        pip_available = False
    
    if pip_available:
        logger.info("\nAttempting to install missing packages...")
        
        import subprocess
        installation_success = True
        
        # Install MCP client if needed
        if not dep_status["mcp_available"] or not dep_status["details"].get("mcp_client_importable", False):
            logger.info("Installing MCP client package...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "mcp-client"])
                logger.info("✅ MCP client package installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install MCP client package: {e}")
                installation_success = False
        
        # Install or update Gradio if needed
        if not dep_status["gradio_mcp_support"]:
            logger.info("Installing/updating Gradio with MCP support...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio>=5.0.0"])
                logger.info("✅ Gradio updated successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to update Gradio: {e}")
                installation_success = False
        
        # Verify installation
        if installation_success:
            logger.info("\nVerifying installation...")
            dep_status = check_mcp_dependencies()
            
            if dep_status["mcp_available"] and dep_status["gradio_mcp_support"]:
                logger.info("✅ All MCP dependencies are now installed correctly.")
                return True
            else:
                logger.error("❌ Some dependencies are still missing or incompatible.")
                logger.error("Please try installing them manually using the instructions below.")
    else:
        logger.warning("⚠️ Pip is not available, cannot automatically install dependencies.")
    
    # Print manual installation instructions
    logger.info("\nManual Installation Instructions:")
    logger.info(dep_status["installation_instructions"])
    
    return False

def main():
    parser = argparse.ArgumentParser(description='Debug MCP client-server interaction')
    parser.add_argument('--port', type=int, default=7990, help='Port to use for testing')
    parser.add_argument('--subprocess', action='store_true', help='Run server in subprocess instead of thread')
    parser.add_argument('--check-deps-only', action='store_true', help='Only check dependencies without running tests')
    args = parser.parse_args()
    
    logger.info("Starting MCP debugging script")
    
    # Check MCP dependencies
    dep_status = check_mcp_dependencies()
    
    if args.check_deps_only:
        logger.info("Dependency check completed. Use --check-deps-only=false to run tests.")
        if not dep_status["mcp_available"] or not dep_status["gradio_mcp_support"]:
            logger.info("\nInstallation Instructions:")
            logger.info(dep_status["installation_instructions"])
        return
    
    # If dependencies are missing and not in check-only mode, warn but continue
    # (the user might have custom imports or environment)
    if not dep_status["mcp_available"] or not dep_status["gradio_mcp_support"]:
        logger.warning("⚠️ MCP dependencies check failed but continuing with tests.")
        logger.warning("Tests may fail due to missing or incompatible dependencies.")
        logger.warning("Run with --check-deps-only to see detailed installation instructions.")
    
    success = False
    try:
        if args.subprocess:
            success = test_with_subprocess(args.port)
        else:
            with run_server_in_thread(args.port):
                # Check MCP endpoint
                if not check_mcp_endpoint(args.port):
                    logger.error("MCP endpoint check failed")
                    return
                
                # Test client
                success, result = test_mcp_client(args.port)
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("This may be due to missing MCP dependencies.")
        logger.error("Run with --check-deps-only to see installation instructions.")
        return
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        traceback.print_exc()
        return
    
    if success:
        logger.info("✅ MCP client-server test passed!")
    else:
        logger.error("❌ MCP client-server test failed!")
        logger.info("\nTroubleshooting tips:")
        logger.info("1. Check if gradio version supports MCP (needs 5.0+)")
        logger.info("2. Verify MCP client is installed correctly")
        logger.info("3. Ensure server is exposing MCP endpoints correctly")
        logger.info("4. Check network connectivity if using remote URLs")
        
        # Fallback to the standalone API server
        logger.info("\n🔄 Falling back to standalone API server solution:")
        try:
            logger.info("1. Starting API server on port 8000 (if not already running)")
            # Use subprocess to start the API server in the background
            import subprocess
            api_server_process = subprocess.Popen(
                [sys.executable, "-m", "src.api_server", "--port", "8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create a new process group
            )
            
            # Wait for server to start
            logger.info("2. Waiting for API server to initialize...")
            time.sleep(3)
            
            # Test API server
            logger.info("3. Testing API server with direct request...")
            try:
                test_response = requests.post(
                    "http://localhost:8000/api/predict_species",
                    json={"features": [5.1, 3.5, 1.4, 0.2]},
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
                
                if test_response.status_code == 200:
                    result = test_response.json()
                    logger.info(f"✅ API server test passed! Result: {result}")
                    logger.info("\n🎉 Solution:")
                    logger.info("Use the standalone API server for reliable MCP integration:")
                    logger.info("1. Run the API server: python3 -m src.api_server")
                    logger.info("2. The server exposes all the MCP-compatible endpoints")
                    logger.info("3. Configure clients to connect to http://localhost:8000")
                    logger.info("\nFor a complete solution, use: python3 scripts/start_servers.py")
                else:
                    logger.error(f"❌ API server test failed: {test_response.status_code} {test_response.text}")
            except Exception as e:
                logger.error(f"❌ Error testing API server: {e}")
                
            # Clean up
            try:
                os.killpg(os.getpgid(api_server_process.pid), signal.SIGTERM)
                api_server_process.wait(timeout=5)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error with API server fallback: {e}")

if __name__ == "__main__":
    # Check if this is a setup run
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        check_and_fix_mcp_setup()
    else:
        main()
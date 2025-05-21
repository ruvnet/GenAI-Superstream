#!/usr/bin/env python3
"""
Start both API and MCP servers for the GenAI-Superstream project.

This script:
1. Starts the standalone API server in a separate process
2. Starts the MCP server that connects to the API server
3. Monitors both servers and handles clean shutdown
"""

import os
import sys
import time
import argparse
import threading
import signal
import subprocess
import logging

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

logger = logging.getLogger("servers")

# Global variables for process tracking
api_server_process = None
mcp_server_process = None
should_exit = threading.Event()

def start_api_server(api_port):
    """Start the standalone API server in a subprocess."""
    global api_server_process
    
    logger.info(f"Starting API server on port {api_port}...")
    
    # Start the API server process
    api_server_process = subprocess.Popen(
        [sys.executable, "-m", "src.api_server", "--port", str(api_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    # Wait for the server to start
    time.sleep(3)
    
    # Check if the server started successfully (process still running)
    if api_server_process.poll() is None:
        logger.info(f"API server started on http://localhost:{api_port}")
        return True
    else:
        stdout, stderr = api_server_process.communicate()
        logger.error("API server failed to start")
        logger.error(f"Output: {stdout.decode('utf-8')}")
        logger.error(f"Error: {stderr.decode('utf-8')}")
        return False

def start_mcp_server(mcp_port, api_port):
    """Start the MCP server in a subprocess."""
    global mcp_server_process
    
    logger.info(f"Starting MCP server on port {mcp_port} (connecting to API on port {api_port})...")
    
    # Set environment variable for API URL
    env = os.environ.copy()
    env["API_SERVER_URL"] = f"http://localhost:{api_port}"
    
    # Start the MCP server process
    mcp_server_process = subprocess.Popen(
        [sys.executable, "-m", "src.server", "--port", str(mcp_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    # Wait for the server to start
    time.sleep(3)
    
    # Check if the server started successfully (process still running)
    if mcp_server_process.poll() is None:
        logger.info(f"MCP server started on http://localhost:{mcp_port}")
        logger.info(f"MCP endpoint available at: http://localhost:{mcp_port}/gradio_api/mcp/sse")
        return True
    else:
        stdout, stderr = mcp_server_process.communicate()
        logger.error("MCP server failed to start")
        logger.error(f"Output: {stdout.decode('utf-8')}")
        logger.error(f"Error: {stderr.decode('utf-8')}")
        return False

def monitor_servers():
    """Monitor the server processes and log their output."""
    while not should_exit.is_set():
        # Check API server status
        if api_server_process and api_server_process.poll() is not None:
            stdout, stderr = api_server_process.communicate()
            logger.error("API server has stopped unexpectedly")
            logger.error(f"Output: {stdout.decode('utf-8')}")
            logger.error(f"Error: {stderr.decode('utf-8')}")
            shutdown_servers()
            break
            
        # Check MCP server status
        if mcp_server_process and mcp_server_process.poll() is not None:
            stdout, stderr = mcp_server_process.communicate()
            logger.error("MCP server has stopped unexpectedly")
            logger.error(f"Output: {stdout.decode('utf-8')}")
            logger.error(f"Error: {stderr.decode('utf-8')}")
            shutdown_servers()
            break
            
        # Sleep to avoid high CPU usage
        time.sleep(1)

def shutdown_servers():
    """Shutdown all servers cleanly."""
    logger.info("Shutting down servers...")
    
    # Stop the API server
    if api_server_process:
        try:
            os.killpg(os.getpgid(api_server_process.pid), signal.SIGTERM)
            api_server_process.wait(timeout=5)
            logger.info("API server stopped")
        except Exception as e:
            logger.error(f"Error stopping API server: {e}")
            try:
                os.killpg(os.getpgid(api_server_process.pid), signal.SIGKILL)
                logger.info("API server forcefully terminated")
            except Exception:
                pass
    
    # Stop the MCP server
    if mcp_server_process:
        try:
            os.killpg(os.getpgid(mcp_server_process.pid), signal.SIGTERM)
            mcp_server_process.wait(timeout=5)
            logger.info("MCP server stopped")
        except Exception as e:
            logger.error(f"Error stopping MCP server: {e}")
            try:
                os.killpg(os.getpgid(mcp_server_process.pid), signal.SIGKILL)
                logger.info("MCP server forcefully terminated")
            except Exception:
                pass
                
    should_exit.set()

def signal_handler(sig, frame):
    """Handle interrupt signals."""
    logger.info("Received interrupt signal, shutting down...")
    shutdown_servers()
    sys.exit(0)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Start API and MCP servers')
    parser.add_argument('--api-port', type=int, default=8000, help='Port for the API server')
    parser.add_argument('--mcp-port', type=int, default=7860, help='Port for the MCP server')
    
    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the API server
        if not start_api_server(args.api_port):
            logger.error("Failed to start API server, exiting")
            return 1
            
        # Start the MCP server
        if not start_mcp_server(args.mcp_port, args.api_port):
            logger.error("Failed to start MCP server, exiting")
            shutdown_servers()
            return 1
            
        logger.info("Both servers started successfully")
        logger.info(f"API server: http://localhost:{args.api_port}")
        logger.info(f"MCP server: http://localhost:{args.mcp_port}")
        logger.info(f"MCP endpoint: http://localhost:{args.mcp_port}/gradio_api/mcp/sse")
        logger.info("Press Ctrl+C to stop the servers")
        
        # Start the monitor thread
        monitor_thread = threading.Thread(target=monitor_servers)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Keep the main thread alive until signal
        while not should_exit.is_set():
            time.sleep(1)
            
        logger.info("Servers stopped")
        return 0
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        shutdown_servers()
        return 1
        
    finally:
        # Ensure servers are shutdown
        shutdown_servers()

if __name__ == "__main__":
    sys.exit(main())
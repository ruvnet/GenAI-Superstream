"""
Script to run the Iris species classifier MCP client.

This script launches a Gradio interface that connects to an MCP server
and allows users to submit feature values for Iris species prediction.
"""

import argparse

from src.client import MCPClient, ClientInterface, ConnectionManager, create_mcp_client
from src.utils import Logger, ConfigManager

def main():
    parser = argparse.ArgumentParser(description='Run Iris MCP Client')
    parser.add_argument('--server', type=str, 
                       default="http://localhost:7860/gradio_api/mcp/sse",
                       help='MCP server URL')
    parser.add_argument('--port', type=int, default=7861,
                       help='Port number for the client interface')
    parser.add_argument('--config', type=str,
                       help='Path to client configuration file')
    parser.add_argument('--share', action='store_true',
                       help='Create a public share link')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = Logger().get_logger()
    
    # Load configuration if provided
    server_url = args.server
    port = args.port
    share = args.share
    
    if args.config:
        config_manager = ConfigManager(args.config)
        client_config = config_manager.get_config('client')
        if client_config:
            server_url = client_config.get('server_url', server_url)
    
    # Create connection manager with retry logic
    logger.info(f"Connecting to MCP server at {server_url}")
    conn_manager = ConnectionManager(server_url, max_retries=3, retry_delay=2)
    success, client = conn_manager.connect_with_retry()
    
    if not success or client is None:
        logger.error(f"Failed to connect to MCP server at {server_url}")
        print(f"ERROR: Failed to connect to MCP server at {server_url}")
        print("Make sure the server is running and the URL is correct.")
        return
    
    # Create and launch client interface
    logger.info("Creating client interface")
    client_interface = ClientInterface(client)
    interface = client_interface.create_interface()
    
    logger.info(f"Launching client interface on port {port}")
    interface.launch(
        server_port=port,
        share=share
    )

if __name__ == "__main__":
    main()
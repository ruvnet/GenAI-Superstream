"""
Script to run the Iris species classifier MCP server.

This script loads a trained model (or trains a new one if not found),
then launches a Gradio MCP server to expose the model's prediction functionality.
"""

import os
import argparse
from pathlib import Path

from src.data import DataLoader
from src.model import ModelTrainer, Predictor
from src.server import MCPServer, ServerConfig, create_mcp_server
from src.utils import Logger, PathManager

def main():
    parser = argparse.ArgumentParser(description='Run Iris MCP Server')
    parser.add_argument('--model', type=str, 
                       help='Path to saved model file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address to bind the server')
    parser.add_argument('--port', type=int, default=7860,
                       help='Port number to bind the server')
    parser.add_argument('--config', type=str,
                       help='Path to server configuration file')
    parser.add_argument('--share', action='store_true',
                       help='Create a public share link')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = Logger().get_logger()
    
    # If config file is provided, use it
    if args.config:
        config = ServerConfig(args.config)
        server_config = config.get_config()
        host = server_config.get("host", args.host)
        port = server_config.get("port", args.port)
        share = server_config.get("share", args.share)
    else:
        host = args.host
        port = args.port
        share = args.share
    
    # Determine model path
    model_path = args.model
    if not model_path:
        # Look for default model in models directory
        models_dir = PathManager.get_model_path()
        default_model = os.path.join(models_dir, "iris_logistic_regression.pkl")
        if os.path.exists(default_model):
            model_path = default_model
            logger.info(f"Using default model at {model_path}")
    
    # Launch server
    logger.info(f"Launching MCP server on {host}:{port}")
    create_mcp_server(
        model_path=model_path,
        host=host,
        port=port
    ).launch(
        server_name=host,
        server_port=port,
        share=share
    )

if __name__ == "__main__":
    main()
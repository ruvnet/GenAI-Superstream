"""
Main entry point for the GenAI-Superstream project.

This script provides command-line functionality to:
1. Train a model
2. Run an MCP server
3. Run an MCP client
4. Run a simplified version (similar to README example)
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='GenAI-Superstream: Exposing scikit-learn models via MCP'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train and save a model')
    train_parser.add_argument('--model-type', type=str, default='logistic_regression',
                           choices=['logistic_regression', 'decision_tree', 'random_forest'],
                           help='Type of model to train')
    train_parser.add_argument('--scaling', type=str, choices=[None, 'standard', 'minmax'],
                           help='Scaling strategy for data preprocessing')
    train_parser.add_argument('--output', type=str, help='Path to save the trained model')
    train_parser.add_argument('--config', type=str, help='Path to configuration file')
    
    # Run server command
    server_parser = subparsers.add_parser('server', help='Run MCP server')
    server_parser.add_argument('--model', type=str, help='Path to saved model file')
    server_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    server_parser.add_argument('--port', type=int, default=7860, help='Port number')
    server_parser.add_argument('--config', type=str, help='Path to server configuration file')
    server_parser.add_argument('--share', action='store_true', help='Create a public share link')
    
    # Run client command
    client_parser = subparsers.add_parser('client', help='Run MCP client')
    client_parser.add_argument('--server', type=str, 
                             default="http://localhost:7860/gradio_api/mcp/sse",
                             help='MCP server URL')
    client_parser.add_argument('--port', type=int, default=7861, help='Port number for client interface')
    client_parser.add_argument('--config', type=str, help='Path to client configuration file')
    client_parser.add_argument('--share', action='store_true', help='Create a public share link')
    
    # Run simplified version (like README example)
    simple_parser = subparsers.add_parser('simple', help='Run simplified version (like README example)')
    simple_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    simple_parser.add_argument('--port', type=int, default=7860, help='Port number')
    simple_parser.add_argument('--share', action='store_true', help='Create a public share link')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'train':
        run_train_script(args)
    elif args.command == 'server':
        run_server_script(args)
    elif args.command == 'client':
        run_client_script(args)
    elif args.command == 'simple':
        run_simple_version(args)
    else:
        parser.print_help()

def run_train_script(args):
    """Run the train model script with the provided arguments."""
    print("Training model...")
    from scripts.train_model import train_and_save_model
    
    model_params = {}
    if args.config:
        from src.utils import ConfigManager
        config_manager = ConfigManager(args.config)
        model_config = config_manager.get_config('model')
        model_type = model_config.get('type', args.model_type)
        model_params = model_config.get('params', {})
    else:
        model_type = args.model_type
    
    # Determine output path
    output_path = args.output
    if not output_path:
        from src.utils import PathManager
        models_dir = PathManager.get_model_path()
        os.makedirs(models_dir, exist_ok=True)
        output_path = os.path.join(models_dir, f"iris_{model_type}.pkl")
    
    # Train and save model
    metrics = train_and_save_model(
        model_type=model_type,
        model_params=model_params,
        scaling=args.scaling,
        output_path=output_path
    )
    
    # Print results
    print("\nModel Training Results:")
    print(f"Model Type: {model_type}")
    print(f"Parameters: {model_params}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Model saved to: {output_path}")

def run_server_script(args):
    """Run the MCP server script with the provided arguments."""
    print("Starting MCP server...")
    from src.server import create_mcp_server
    
    # Parse configuration if provided
    host = args.host
    port = args.port
    share = args.share
    
    if args.config:
        from src.utils import ConfigManager
        from src.server import ServerConfig
        config = ServerConfig(args.config)
        server_config = config.get_config()
        host = server_config.get("host", args.host)
        port = server_config.get("port", args.port)
        share = server_config.get("share", args.share)
    
    # Launch server
    print(f"MCP Server will be available at http://{host}:{port}/gradio_api/mcp/sse")
    create_mcp_server(
        model_path=args.model,
        host=host,
        port=port
    ).launch(
        server_name=host,
        server_port=port,
        share=share
    )

def run_client_script(args):
    """Run the MCP client script with the provided arguments."""
    print("Starting MCP client...")
    from src.client import ConnectionManager, ClientInterface
    
    # Parse configuration if provided
    server_url = args.server
    port = args.port
    share = args.share
    
    if args.config:
        from src.utils import ConfigManager
        config_manager = ConfigManager(args.config)
        client_config = config_manager.get_config('client')
        if client_config:
            server_url = client_config.get('server_url', server_url)
    
    # Connect to server
    print(f"Connecting to MCP server at {server_url}")
    conn_manager = ConnectionManager(server_url, max_retries=3, retry_delay=2)
    success, client = conn_manager.connect_with_retry()
    
    if not success or client is None:
        print(f"ERROR: Failed to connect to MCP server at {server_url}")
        print("Make sure the server is running and the URL is correct.")
        return
    
    # Launch client interface
    print(f"Starting client interface on port {port}")
    client_interface = ClientInterface(client)
    interface = client_interface.create_interface()
    interface.launch(server_port=port, share=share)

def run_simple_version(args):
    """Run a simplified version similar to the README example."""
    print("Running simplified version (like README example)...")
    
    # Import required libraries
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    import gradio as gr
    
    # Load data and train
    iris = load_iris()
    clf = LogisticRegression(max_iter=200)
    clf.fit(iris.data, iris.target)
    
    def predict_species(features: list) -> dict:
        """
        Predicts the Iris species given a list of four features:
        [sepal_length, sepal_width, petal_length, petal_width].
        Returns a dict with class probabilities.
        """
        probs = clf.predict_proba([features])[0]
        classes = iris.target_names.tolist()
        return dict(zip(classes, probs))
    
    with gr.Blocks() as demo:
        gr.Markdown("# Iris Species Predictor (Simple Version)")
        gr.Markdown("Enter 4 comma-separated values for: sepal length, sepal width, petal length, petal width")
        
        inp = gr.Textbox(label="Comma-separated features")
        out = gr.JSON(label="Class probabilities")
        
        def wrapper(text):
            features = list(map(float, text.split(",")))
            return predict_species(features)
        
        inp.submit(wrapper, inp, out)
        
        # Add examples
        examples = [
            ["5.1, 3.5, 1.4, 0.2"],  # Setosa
            ["7.0, 3.2, 4.7, 1.4"],  # Versicolor
            ["6.3, 3.3, 6.0, 2.5"]   # Virginica
        ]
        gr.Examples(examples=examples, inputs=inp)
    
    print(f"Launching simple MCP server on {args.host}:{args.port}")
    demo.launch(
        server_name=args.host, 
        server_port=args.port, 
        mcp_server=True,
        share=args.share
    )

if __name__ == "__main__":
    main()
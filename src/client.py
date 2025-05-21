"""
Client module for the GenAI-Superstream project.

This module implements a client that connects to the MCP server
and provides methods to call the prediction tool.
"""

import os
import time
import json
import gradio as gr
from mcp import client as mcp_client

from src.utils import Logger, ConfigManager, InputValidator, FormatConverter

logger = Logger().get_logger()

class MCPClient:
    """
    Client for interacting with the MCP server.
    """
    
    def __init__(self, server_url):
        """
        Initialize the MCPClient.
        
        Args:
            server_url (str): URL of the MCP server endpoint
        """
        self.server_url = server_url
        self.client = None
        self.connected = False
        
    def connect(self):
        """
        Connect to the MCP server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to MCP server at {self.server_url}")
            # Create the MCP client instance with proper error handling
            try:
                # First approach: try using the module directly as a callable
                self.client = mcp_client(self.server_url)
                logger.info("Created MCP client using module callable")
                
                # Verify connection by listing available tools
                tools = self.client.list_tools()
                logger.info(f"Available MCP tools: {tools}")
                
                # Check if predict_species tool is available
                if "predict_species" in tools:
                    logger.info("predict_species tool is available")
                else:
                    logger.warning("predict_species tool not found in available tools")
                
                self.connected = True
                logger.info("Connected to MCP server")
                return True
            except Exception as e:
                logger.error(f"Error creating client as callable: {e}")
                logger.info("Trying alternative client initialization")
                
                # If the first approach fails, try importing Client directly
                try:
                    # Import the Client class directly
                    from mcp.client import Client
                    self.client = Client(self.server_url)
                    self.connected = True
                    logger.info("Connected to MCP server using Client class")
                    return True
                except Exception as e2:
                    logger.error(f"Error with Client class approach: {e2}")
                    raise RuntimeError(f"All client initialization methods failed")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            self.connected = False
            return False
        
    def call_prediction_tool(self, features):
        """
        Call the prediction tool on the MCP server.
        
        Args:
            features (list): List of feature values
            
        Returns:
            dict: Prediction result from the server
        """
        # Check if connected, connect if not
        if not self.connected or self.client is None:
            success = self.connect()
            if not success:
                raise ConnectionError("Failed to connect to MCP server")
                
        # Validate the features input
        features = self.validate_features(features)
        
        try:
            # Call the "predict_species" tool with the features
            logger.info(f"Calling predict_species tool with features: {features}")
            response = self.client.call(
                tool="predict_species",
                input={"features": features}
            )
            
            logger.info(f"Received response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error calling prediction tool: {e}")
            raise
        
    def validate_features(self, features):
        """
        Validate the feature input for prediction.
        
        Args:
            features: The features to validate
            
        Returns:
            list: Validated and formatted feature list
        """
        # Check if features is a string or list
        if isinstance(features, str):
            # Split by commas and convert to float
            try:
                features = [float(x.strip()) for x in features.split(',')]
            except ValueError:
                raise ValueError("Invalid feature format. Expected comma-separated floats.")
        
        # Check if it's a list and convert elements to float
        elif isinstance(features, (list, tuple)):
            try:
                features = [float(x) for x in features]
            except (ValueError, TypeError):
                raise ValueError("Invalid feature values. All values must be convertible to float.")
        else:
            raise TypeError(f"Unsupported feature type: {type(features)}")
            
        # Check if resulting list has exactly 4 elements
        if len(features) != 4:
            raise ValueError(f"Features must have exactly 4 elements, got {len(features)}")
            
        return features
        
    def disconnect(self):
        """
        Disconnect from the MCP server.
        """
        logger.info("Disconnecting from MCP server")
        # The MCP Client doesn't have a specific disconnect method
        # Reset our client state
        self.client = None
        self.connected = False


class ClientInterface:
    """
    Gradio interface for interacting with the MCP client.
    """
    
    def __init__(self, client):
        """
        Initialize the ClientInterface.
        
        Args:
            client: MCPClient instance
        """
        self.client = client
        self.interface = None
        
    def create_interface(self):
        """
        Create the Gradio interface for the client.
        
        Returns:
            gradio.Interface: The created interface
        """
        logger.info("Creating Gradio interface for MCP client")
        
        # Define the ask_model function that calls the client's prediction tool
        def ask_model(prompt):
            try:
                # Parse features from the prompt
                features = self.client.validate_features(prompt)
                
                # Call the MCP tool and get the response
                response = self.client.call_prediction_tool(features)
                
                # Add the predicted class
                if "error" not in response:
                    max_class = max(response.items(), key=lambda x: x[1])[0]
                    formatted_response = {
                        "probabilities": response,
                        "predicted_class": max_class,
                        "confidence": response[max_class]
                    }
                    return formatted_response
                return response
                
            except Exception as e:
                logger.error(f"Error in ask_model: {e}")
                return {"error": str(e)}
        
        # Create a Gradio Interface
        self.interface = gr.Interface(
            fn=ask_model,
            inputs=gr.Textbox(label="Features (comma-separated)"),
            outputs=gr.JSON(label="Prediction"),
            title="Iris Species Predictor Client",
            description="Enter 4 comma-separated values for: sepal length, sepal width, petal length, petal width",
            examples=[
                ["5.1, 3.5, 1.4, 0.2"],  # Setosa
                ["7.0, 3.2, 4.7, 1.4"],  # Versicolor
                ["6.3, 3.3, 6.0, 2.5"]   # Virginica
            ]
        )
        
        return self.interface
        
    def launch(self, **kwargs):
        """
        Launch the client interface.
        
        Args:
            **kwargs: Additional parameters for Gradio launch
            
        Returns:
            gradio.Interface: The launched interface
        """
        # If interface is not created, call create_interface()
        if not self.interface:
            self.create_interface()
            
        logger.info(f"Launching client interface with parameters: {kwargs}")
        return self.interface.launch(**kwargs)


class ClientConfig:
    """
    Manages the configuration for the MCP client.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the ClientConfig.
        
        Args:
            config_file (str, optional): Path to a configuration file
        """
        self.config = self._default_config()
        if config_file:
            self.load_config(config_file)
        
    def _default_config(self):
        """
        Provide the default client configuration.
        
        Returns:
            dict: Default configuration
        """
        return {
            "server_url": "http://localhost:7860/gradio_api/mcp/sse",
            "timeout": 30,
            "retry_attempts": 3,
            "retry_delay": 2
        }
        
    def load_config(self, config_file):
        """
        Load configuration from a file.
        
        Args:
            config_file (str): Path to the configuration file
        """
        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            return
            
        try:
            # Determine file type and load
            if config_file.endswith('.json'):
                import json
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
            elif config_file.endswith(('.yaml', '.yml')):
                import yaml
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported config file format: {config_file}")
                return
                
            # Update config with loaded values
            self.config.update(loaded_config)
            logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
        
    def get_config(self):
        """
        Get the current configuration.
        
        Returns:
            dict: Current configuration
        """
        return self.config
        
    def update_config(self, updates):
        """
        Update the configuration with new values.
        
        Args:
            updates (dict): Dictionary of configuration updates
        """
        self.config.update(updates)
        logger.info("Client configuration updated")


class ConnectionManager:
    """
    Manages the connection to the MCP server with retry logic.
    """
    
    def __init__(self, server_url, max_retries=3, retry_delay=2):
        """
        Initialize the ConnectionManager.
        
        Args:
            server_url (str): URL of the MCP server endpoint
            max_retries (int): Maximum number of connection attempts
            retry_delay (int): Delay between retry attempts in seconds
        """
        self.server_url = server_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def connect_with_retry(self):
        """
        Attempt to connect to the server with retry logic.
        
        Returns:
            tuple: (success, client) - success is a boolean, client is the MCP client or None
        """
        retries = 0
        client = MCPClient(self.server_url)
        
        while retries < self.max_retries:
            try:
                logger.info(f"Connection attempt {retries + 1}/{self.max_retries}")
                success = client.connect()
                
                if success:
                    logger.info("Connection successful")
                    return True, client
                    
                retries += 1
                if retries < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"Connection error: {e}")
                retries += 1
                if retries < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
        
        logger.error(f"Failed to connect after {self.max_retries} attempts")
        return False, None


def create_mcp_client(server_url=None, config_file=None):
    """
    Create and launch a complete MCP client for the Iris classifier.
    
    Args:
        server_url (str, optional): URL of the MCP server endpoint
        config_file (str, optional): Path to a configuration file
        
    Returns:
        gradio.Interface: The launched client interface
    """
    # Load configuration if provided
    if config_file:
        client_config = ClientConfig(config_file)
        config = client_config.get_config()
        server_url = server_url or config.get("server_url")
        max_retries = config.get("retry_attempts", 3)
        retry_delay = config.get("retry_delay", 2)
    else:
        server_url = server_url or "http://localhost:7860/gradio_api/mcp/sse"
        max_retries = 3
        retry_delay = 2
    
    # Create connection manager and connect with retry logic
    conn_manager = ConnectionManager(server_url, max_retries, retry_delay)
    success, client = conn_manager.connect_with_retry()
    
    if not success or client is None:
        raise ConnectionError(f"Failed to connect to MCP server at {server_url}")
    
    # Create and launch the client interface
    client_interface = ClientInterface(client)
    interface = client_interface.create_interface()
    
    return interface.launch()


# Example usage if this module is run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Iris MCP Client')
    parser.add_argument('--server', type=str, 
                      default="http://localhost:7860/gradio_api/mcp/sse",
                      help='MCP server URL')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    # Launch the client
    create_mcp_client(args.server, args.config)
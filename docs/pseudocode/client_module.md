# Client Module Pseudocode

This document outlines the pseudocode for the client components of the GenAI-Superstream project, which interact with the MCP server.

## MCPClient Class

```python
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
        # TEST: Verify client can connect to server successfully
        # Create an MCP Client instance using the server URL
        # Attempt to establish a connection
        # Handle any connection errors
        # Set connected flag to True if successful
        # Return connection status
        
    def call_prediction_tool(self, features):
        """
        Call the prediction tool on the MCP server.
        
        Args:
            features (list): List of feature values
            
        Returns:
            dict: Prediction result from the server
        """
        # TEST: Verify tool calling returns expected response format
        # Check if connected, connect if not
        # Validate the features input
        # Format the features for the MCP request
        # Call the "predict_species" tool with the features
        # Handle any errors that occur during the call
        # Return the response from the server
        
    def validate_features(self, features):
        """
        Validate the feature input for prediction.
        
        Args:
            features: The features to validate
            
        Returns:
            list: Validated and formatted feature list
        """
        # TEST: Verify validation catches invalid inputs
        # Check if features is a string or list
        # If string, split by commas and convert to float
        # If list, check length and convert elements to float
        # Check if resulting list has exactly 4 elements
        # Return the validated feature list
        # Raise ValueError if validation fails
        
    def disconnect(self):
        """
        Disconnect from the MCP server.
        """
        # TEST: Verify client can disconnect cleanly
        # Close the client connection if it exists
        # Set connected flag to False
```

## ClientInterface Class

```python
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
        # TEST: Verify client interface is created with expected components
        # Import gradio
        # Define the ask_model function that:
        #   - Takes a prompt (comma-separated feature values)
        #   - Calls the client's prediction tool
        #   - Returns the prediction result
        # Create a Gradio Interface with:
        #   - Input: Textbox for feature input
        #   - Output: JSON component for prediction results
        #   - Function: ask_model function
        # Return the created interface
        
    def launch(self, **kwargs):
        """
        Launch the client interface.
        
        Args:
            **kwargs: Additional parameters for Gradio launch
            
        Returns:
            gradio.Interface: The launched interface
        """
        # TEST: Verify client interface launches successfully
        # If interface is not created, call create_interface()
        # Launch the interface with the provided kwargs
        # Return the launched interface
```

## ClientConfig Class

```python
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
        # TEST: Verify default config has all required settings
        # Return default configuration dictionary with:
        #   - server_url: "http://localhost:7860/gradio_api/mcp/sse"
        #   - timeout: 30
        #   - retry_attempts: 3
        #   - retry_delay: 2
        
    def load_config(self, config_file):
        """
        Load configuration from a file.
        
        Args:
            config_file (str): Path to the configuration file
        """
        # TEST: Verify config loading correctly overrides defaults
        # Load configuration from file (JSON, YAML, etc.)
        # Update self.config with loaded values
        # Validate the loaded configuration
        
    def get_config(self):
        """
        Get the current configuration.
        
        Returns:
            dict: Current configuration
        """
        # Return the current configuration dictionary
        
    def update_config(self, updates):
        """
        Update the configuration with new values.
        
        Args:
            updates (dict): Dictionary of configuration updates
        """
        # TEST: Verify config updates modify settings correctly
        # Update self.config with the values in updates
        # Validate the updated configuration
```

## ConnectionManager Class

```python
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
        # TEST: Verify retry logic handles connection failures appropriately
        # Initialize retry counter and success flag
        # Create a new MCPClient instance
        # Loop until max_retries is reached:
        #   - Attempt to connect using client.connect()
        #   - If successful, return (True, client)
        #   - If failed, increment retry counter
        #   - If retries < max_retries, wait for retry_delay seconds
        #   - Otherwise, return (False, None)
```

## Usage Example

```python
# Example usage of the client module
import gradio as gr
from mcp import Client

# Create an MCP client
server_url = "http://localhost:7860/gradio_api/mcp/sse"
mcp_client = MCPClient(server_url)
mcp_client.connect()

# Define the function to call the MCP tool
def ask_model(prompt):
    # Parse the input features from the prompt
    features = [float(x.strip()) for x in prompt.split(",")]
    
    # Call the MCP tool and get the response
    response = mcp_client.call_prediction_tool(features)
    
    # Return the response
    return response

# Create a Gradio interface
iface = gr.Interface(
    fn=ask_model,
    inputs=gr.Textbox(label="Features (comma-separated)"),
    outputs=gr.JSON(label="Prediction")
)

# Launch the interface
iface.launch()
```

## Complete MCP Client Implementation

```python
def create_mcp_client(server_url="http://localhost:7860/gradio_api/mcp/sse"):
    """
    Create and launch a complete MCP client for the Iris classifier.
    
    Args:
        server_url (str): URL of the MCP server endpoint
        
    Returns:
        gradio.Interface: The launched client interface
    """
    # Import required modules
    # Create an MCP Client instance
    # Connect to the server
    
    # Define the function to call the MCP tool
    def ask_model(prompt):
        # Parse the input features from the prompt
        # Call the MCP tool and get the response
        # Return the response
    
    # Create and launch a Gradio interface
    # Return the interface for reference or testing
```

## Interfaces

The Client Module exposes the following interfaces:

1. `MCPClient.connect()`: Connects to the MCP server
2. `MCPClient.call_prediction_tool()`: Calls the prediction tool on the server
3. `ClientInterface.create_interface()`: Creates the Gradio interface for the client
4. `ClientInterface.launch()`: Launches the client interface
5. `ClientConfig.get_config()`: Gets the current client configuration
6. `ConnectionManager.connect_with_retry()`: Manages connection with retry logic
7. `create_mcp_client()`: Creates and launches a complete MCP client

## Error Handling

- Connection errors are caught and reported with helpful messages
- Tool calling errors are handled gracefully
- Input validation errors provide clear explanations
- Retry logic helps handle temporary connection issues

## Testing Anchors

- TEST: Verify client can connect to server successfully
- TEST: Verify tool calling returns expected response format
- TEST: Verify validation catches invalid inputs
- TEST: Verify client can disconnect cleanly
- TEST: Verify client interface is created with expected components
- TEST: Verify client interface launches successfully
- TEST: Verify default config has all required settings
- TEST: Verify config loading correctly overrides defaults
- TEST: Verify config updates modify settings correctly
- TEST: Verify retry logic handles connection failures appropriately
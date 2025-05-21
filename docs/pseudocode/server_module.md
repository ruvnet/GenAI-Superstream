# Server Module Pseudocode

This document outlines the pseudocode for the server components of the GenAI-Superstream project, which expose the model via MCP using Gradio.

## GradioInterface Class

```python
class GradioInterface:
    """
    Creates and manages the Gradio interface for the model.
    """
    
    def __init__(self, predictor):
        """
        Initialize the GradioInterface.
        
        Args:
            predictor: Predictor instance for making predictions
        """
        self.predictor = predictor
        self.blocks = None
        
    def create_interface(self):
        """
        Create the Gradio Blocks interface.
        
        Returns:
            gradio.Blocks: The created Blocks interface
        """
        # TEST: Verify interface is created with expected components
        # Import gradio
        # Create a Blocks interface with:
        #   - A Textbox for input features
        #   - A JSON component for output results
        # Define a wrapper function that:
        #   - Parses the input text to extract features
        #   - Calls the predictor to make a prediction
        #   - Returns the prediction result
        # Set up the submission event to call the wrapper function
        # Return the Blocks interface
        
    def parse_input(self, text):
        """
        Parse the input text into a feature list.
        
        Args:
            text (str): Comma-separated feature values
            
        Returns:
            list: List of float feature values
        """
        # TEST: Verify input parsing correctly handles valid and invalid inputs
        # Split the text by commas
        # Convert each value to float
        # Return the list of float values
        # Handle potential errors in conversion
        
    def format_output(self, prediction):
        """
        Format the prediction result for display.
        
        Args:
            prediction (dict): Dictionary of class probabilities
            
        Returns:
            dict: Formatted prediction result
        """
        # TEST: Verify output formatting provides clean, readable results
        # Return the prediction dictionary directly
        # Or format it further if needed for display purposes
```

## MCPServer Class

```python
class MCPServer:
    """
    Manages the MCP server configuration and launch.
    """
    
    def __init__(self, predictor, host="0.0.0.0", port=7860):
        """
        Initialize the MCPServer.
        
        Args:
            predictor: Predictor instance for making predictions
            host (str): Host address to bind the server
            port (int): Port number to bind the server
        """
        self.predictor = predictor
        self.host = host
        self.port = port
        self.interface = None
        self.server = None
        
    def setup(self):
        """
        Set up the MCP server.
        
        Returns:
            gradio.Blocks: The configured Blocks interface
        """
        # TEST: Verify server setup configures interface correctly
        # Create a GradioInterface instance with the predictor
        # Create the Blocks interface
        # Store the interface for later use
        # Return the configured interface
        
    def expose_prediction_tool(self):
        """
        Expose the prediction function as an MCP tool.
        
        This function is automatically exposed via the docstring when
        mcp_server=True is set in launch().
        
        Returns:
            function: The prediction function exposed as an MCP tool
        """
        # TEST: Verify MCP tool is exposed with correct docstring and signature
        # Define a prediction function with appropriate docstring:
        # """
        # Predicts the Iris species given a list of four features:
        # [sepal_length, sepal_width, petal_length, petal_width].
        # Returns a dict with class probabilities.
        # """
        # The function should call self.predictor.predict_species()
        # Return the prediction function
        
    def launch(self, **kwargs):
        """
        Launch the MCP server.
        
        Args:
            **kwargs: Additional parameters for Gradio launch
            
        Returns:
            gradio.Blocks: The launched Blocks interface
        """
        # TEST: Verify server launches with MCP capabilities enabled
        # If interface is not set up, call setup()
        # Set default launch parameters:
        #   - server_name = self.host
        #   - server_port = self.port
        #   - mcp_server = True
        # Override defaults with any provided kwargs
        # Launch the interface with these parameters
        # Return the launched interface
        
    def shutdown(self):
        """
        Shutdown the MCP server.
        """
        # TEST: Verify server can be cleanly shut down
        # Close the server if it's running
```

## ServerConfig Class

```python
class ServerConfig:
    """
    Manages the configuration for the MCP server.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the ServerConfig.
        
        Args:
            config_file (str, optional): Path to a configuration file
        """
        self.config = self._default_config()
        if config_file:
            self.load_config(config_file)
        
    def _default_config(self):
        """
        Provide the default server configuration.
        
        Returns:
            dict: Default configuration
        """
        # TEST: Verify default config contains all required settings
        # Return default configuration dictionary with:
        #   - host: "0.0.0.0"
        #   - port: 7860
        #   - share: False
        #   - auth: None
        #   - ssl_verify: True
        #   - mcp_server: True
        
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
        # TEST: Verify config updates correctly modify settings
        # Update self.config with the values in updates
        # Validate the updated configuration
```

## Usage Example

```python
# Example usage of the server module
from data_module import DataLoader
from model_module import ModelTrainer, Predictor

# Load data
data_loader = DataLoader()
data, target, feature_names, target_names = data_loader.load_iris_dataset()

# Train model
trainer = ModelTrainer()
model = trainer.train(data, target)

# Create predictor
predictor = Predictor(model, target_names)

# Create MCP server
server = MCPServer(predictor, host="0.0.0.0", port=7860)

# Launch server
server.launch(share=False)

# This will start both the web UI and MCP server
# The MCP endpoint will be available at: http://0.0.0.0:7860/gradio_api/mcp/sse
```

## Complete MCP Server Implementation

```python
def create_mcp_server():
    """
    Create and launch a complete MCP server using the Iris classifier.
    """
    # Import required modules
    # Load the Iris dataset
    # Train the model
    # Create the predictor
    
    # Define the MCP-exposed prediction function
    def predict_species(features):
        """
        Predicts the Iris species given a list of four features:
        [sepal_length, sepal_width, petal_length, petal_width].
        Returns a dict with class probabilities.
        """
        # Call the predictor to make a prediction
        # Return the prediction result
    
    # Create the Gradio interface
    # Launch the interface with mcp_server=True
    
    # Return the interface for reference or testing
```

## Interfaces

The Server Module exposes the following interfaces:

1. `GradioInterface.create_interface()`: Creates the Gradio Blocks interface
2. `MCPServer.setup()`: Sets up the MCP server
3. `MCPServer.launch()`: Launches the MCP server
4. `MCPServer.expose_prediction_tool()`: Exposes the prediction function as an MCP tool
5. `ServerConfig.get_config()`: Gets the current server configuration
6. `create_mcp_server()`: Creates and launches a complete MCP server

## Error Handling

- Input parsing errors are caught and reported
- Server configuration errors are validated and reported
- Server launch failures are handled gracefully
- Network connection issues are reported with helpful messages

## Security Considerations

- In production, add authentication to protect the MCP endpoint
- Validate inputs thoroughly to prevent injection attacks
- Consider rate limiting for high-traffic scenarios
- Use HTTPS in production environments

## Testing Anchors

- TEST: Verify interface is created with expected components
- TEST: Verify input parsing correctly handles valid and invalid inputs
- TEST: Verify output formatting provides clean, readable results
- TEST: Verify server setup configures interface correctly
- TEST: Verify MCP tool is exposed with correct docstring and signature
- TEST: Verify server launches with MCP capabilities enabled
- TEST: Verify server can be cleanly shut down
- TEST: Verify default config contains all required settings
- TEST: Verify config loading correctly overrides defaults
- TEST: Verify config updates correctly modify settings
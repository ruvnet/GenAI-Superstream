# Utils Module Pseudocode

This document outlines the pseudocode for the utilities components of the GenAI-Superstream project, providing shared functionality across modules.

## Logger Class

```python
class Logger:
    """
    Provides logging functionality for the application.
    """
    
    def __init__(self, log_level="INFO", log_file=None):
        """
        Initialize the Logger.
        
        Args:
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file (str, optional): Path to log file
        """
        self.log_level = log_level
        self.log_file = log_file
        self.logger = None
        
    def setup(self):
        """
        Set up the logger with the specified configuration.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        # TEST: Verify logger is configured with correct settings
        # Import the logging module
        # Create a logger instance
        # Set the log level
        # Create handlers (console and file if log_file is specified)
        # Set formatter for handlers
        # Add handlers to logger
        # Return the configured logger
        
    def get_logger(self):
        """
        Get the configured logger instance.
        
        Returns:
            logging.Logger: The logger instance
        """
        # If logger is not set up, call setup()
        # Return the logger instance
        
    def debug(self, message):
        """Log a debug message."""
        # Get the logger and call logger.debug()
        
    def info(self, message):
        """Log an info message."""
        # Get the logger and call logger.info()
        
    def warning(self, message):
        """Log a warning message."""
        # Get the logger and call logger.warning()
        
    def error(self, message):
        """Log an error message."""
        # Get the logger and call logger.error()
        
    def critical(self, message):
        """Log a critical message."""
        # Get the logger and call logger.critical()
```

## ConfigManager Class

```python
class ConfigManager:
    """
    Manages configuration for the application.
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls, config_file=None):
        """
        Create or return the singleton instance.
        
        Args:
            config_file (str, optional): Path to configuration file
        """
        # TEST: Verify singleton pattern works correctly
        # If _instance is None, create a new instance
        # Otherwise, return the existing instance
        
    def __init__(self, config_file=None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_file (str, optional): Path to configuration file
        """
        # Only initialize if this is a new instance
        if not hasattr(self, 'initialized'):
            self.config_file = config_file
            self.config = self._default_config()
            if config_file:
                self.load_config(config_file)
            self.initialized = True
        
    def _default_config(self):
        """
        Provide the default configuration.
        
        Returns:
            dict: Default configuration
        """
        # TEST: Verify default config has expected values
        # Return default configuration dictionary with:
        #   - model: {"type": "logistic_regression", "params": {"max_iter": 200}}
        #   - server: {"host": "0.0.0.0", "port": 7860, "mcp_server": True}
        #   - client: {"server_url": "http://localhost:7860/gradio_api/mcp/sse"}
        #   - logging: {"level": "INFO", "file": None}
        
    def load_config(self, config_file):
        """
        Load configuration from a file.
        
        Args:
            config_file (str): Path to the configuration file
        """
        # TEST: Verify config loading overrides defaults correctly
        # Determine file format from extension (json, yaml, etc.)
        # Load configuration from file
        # Update self.config with loaded values
        # Validate the loaded configuration
        
    def get_config(self, section=None):
        """
        Get the current configuration.
        
        Args:
            section (str, optional): Configuration section to return
            
        Returns:
            dict: Current configuration or section
        """
        # If section is specified, return that section
        # Otherwise, return the entire configuration
        
    def update_config(self, updates, section=None):
        """
        Update the configuration with new values.
        
        Args:
            updates (dict): Dictionary of configuration updates
            section (str, optional): Section to update
        """
        # TEST: Verify config updates modify settings correctly
        # If section is specified, update that section
        # Otherwise, update the entire configuration
        # Validate the updated configuration
        
    def save_config(self, config_file=None):
        """
        Save the current configuration to a file.
        
        Args:
            config_file (str, optional): Path to save the configuration
        """
        # TEST: Verify config can be saved and loaded with identical values
        # If config_file is not specified, use self.config_file
        # Determine file format from extension
        # Save configuration to file
```

## InputValidator Class

```python
class InputValidator:
    """
    Provides input validation functionality.
    """
    
    @staticmethod
    def validate_feature_vector(features):
        """
        Validate a feature vector for Iris prediction.
        
        Args:
            features: The feature vector to validate
            
        Returns:
            bool: True if valid, False otherwise
            str: Error message if invalid, None if valid
        """
        # TEST: Verify validation correctly identifies valid and invalid inputs
        # Check if features is a list or array-like
        # Check if features has exactly 4 elements
        # Check if all elements can be converted to float
        # Check if values are within reasonable ranges for Iris dataset
        # Return (True, None) if valid
        # Return (False, error_message) if invalid
        
    @staticmethod
    def validate_model_params(model_type, params):
        """
        Validate model parameters for the specified model type.
        
        Args:
            model_type (str): Type of model
            params (dict): Model parameters
            
        Returns:
            bool: True if valid, False otherwise
            str: Error message if invalid, None if valid
        """
        # TEST: Verify validation catches invalid model parameters
        # Check if model_type is supported
        # Check if params contains valid keys for the model type
        # Check if parameter values are of correct types
        # Check if parameter values are within valid ranges
        # Return (True, None) if valid
        # Return (False, error_message) if invalid
        
    @staticmethod
    def validate_server_config(config):
        """
        Validate server configuration.
        
        Args:
            config (dict): Server configuration
            
        Returns:
            bool: True if valid, False otherwise
            str: Error message if invalid, None if valid
        """
        # TEST: Verify validation catches invalid server configuration
        # Check if required keys are present (host, port)
        # Check if values are of correct types
        # Check if port is within valid range
        # Return (True, None) if valid
        # Return (False, error_message) if invalid
```

## FormatConverter Class

```python
class FormatConverter:
    """
    Provides format conversion utilities.
    """
    
    @staticmethod
    def features_to_array(features):
        """
        Convert features to a numpy array.
        
        Args:
            features: Features in various formats (list, string, etc.)
            
        Returns:
            numpy.ndarray: Features as a numpy array
        """
        # TEST: Verify conversion handles different input formats correctly
        # If features is a string, split by commas and convert to float
        # If features is a list, convert to numpy array
        # If features is already a numpy array, return as is
        # Reshape to ensure correct dimensions for scikit-learn
        # Return the converted array
        
    @staticmethod
    def dict_to_json(data):
        """
        Convert a dictionary to a JSON string.
        
        Args:
            data (dict): Dictionary to convert
            
        Returns:
            str: JSON string
        """
        # Import json
        # Convert dictionary to JSON string
        # Handle any serialization errors
        # Return the JSON string
        
    @staticmethod
    def json_to_dict(json_str):
        """
        Convert a JSON string to a dictionary.
        
        Args:
            json_str (str): JSON string to convert
            
        Returns:
            dict: Converted dictionary
        """
        # Import json
        # Convert JSON string to dictionary
        # Handle any parsing errors
        # Return the dictionary
```

## PathManager Class

```python
class PathManager:
    """
    Manages file paths for the application.
    """
    
    @staticmethod
    def get_project_root():
        """
        Get the project root directory.
        
        Returns:
            str: Path to the project root directory
        """
        # Determine the project root directory
        # Return the path
        
    @staticmethod
    def get_config_path(filename=None):
        """
        Get the path to a configuration file.
        
        Args:
            filename (str, optional): Name of the configuration file
            
        Returns:
            str: Path to the configuration file
        """
        # Get the project root
        # Join with 'config' directory
        # If filename is provided, join with filename
        # Return the path
        
    @staticmethod
    def get_model_path(filename=None):
        """
        Get the path to a model file.
        
        Args:
            filename (str, optional): Name of the model file
            
        Returns:
            str: Path to the model file
        """
        # Get the project root
        # Join with 'models' directory
        # If filename is provided, join with filename
        # Return the path
        
    @staticmethod
    def get_log_path(filename=None):
        """
        Get the path to a log file.
        
        Args:
            filename (str, optional): Name of the log file
            
        Returns:
            str: Path to the log file
        """
        # Get the project root
        # Join with 'logs' directory
        # If filename is provided, join with filename
        # Return the path
        
    @staticmethod
    def ensure_directory_exists(path):
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path (str): Path to the directory
            
        Returns:
            bool: True if directory exists or was created, False otherwise
        """
        # Check if the directory exists
        # If not, create it and any necessary parent directories
        # Return True if successful, False otherwise
```

## Usage Example

```python
# Example usage of the utils module

# Configure logging
logger = Logger(log_level="INFO", log_file="app.log")
log = logger.get_logger()
log.info("Application starting")

# Load configuration
config_manager = ConfigManager("config.yaml")
model_config = config_manager.get_config("model")
server_config = config_manager.get_config("server")

# Validate inputs
features = [5.1, 3.5, 1.4, 0.2]
is_valid, error = InputValidator.validate_feature_vector(features)
if not is_valid:
    log.error(f"Invalid features: {error}")
else:
    # Convert features for model input
    features_array = FormatConverter.features_to_array(features)
    
    # Load or save a model
    model_path = PathManager.get_model_path("iris_model.pkl")
    PathManager.ensure_directory_exists(os.path.dirname(model_path))
```

## Interfaces

The Utils Module exposes the following interfaces:

1. `Logger.get_logger()`: Gets a configured logger instance
2. `ConfigManager.get_config()`: Gets the current configuration
3. `InputValidator.validate_feature_vector()`: Validates feature inputs
4. `FormatConverter.features_to_array()`: Converts features to numpy array
5. `PathManager.get_model_path()`: Gets the path to a model file

## Error Handling

- Configuration errors are validated and reported
- Input validation provides detailed error messages
- Format conversion handles different input types gracefully
- Path operations check for existence and permissions

## Testing Anchors

- TEST: Verify logger is configured with correct settings
- TEST: Verify singleton pattern works correctly
- TEST: Verify default config has expected values
- TEST: Verify config loading overrides defaults correctly
- TEST: Verify config updates modify settings correctly
- TEST: Verify config can be saved and loaded with identical values
- TEST: Verify validation correctly identifies valid and invalid inputs
- TEST: Verify validation catches invalid model parameters
- TEST: Verify validation catches invalid server configuration
- TEST: Verify conversion handles different input formats correctly
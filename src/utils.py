"""
Utility functions for the GenAI-Superstream project.

This module provides shared functionality used across other modules
including logging, configuration management, input validation,
format conversion, and path management.
"""

import os
import logging
import json
import numpy as np
from pathlib import Path

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
        # Import the logging module
        logger = logging.getLogger("genai-superstream")
        
        # Set the log level
        level = getattr(logging, self.log_level.upper())
        logger.setLevel(level)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Set formatter for handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add file handler if log_file is specified
        if self.log_file:
            # Ensure directory exists
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Add console handler to logger
        logger.addHandler(console_handler)
        
        self.logger = logger
        return logger
        
    def get_logger(self):
        """
        Get the configured logger instance.
        
        Returns:
            logging.Logger: The logger instance
        """
        if not self.logger:
            self.setup()
        return self.logger
        
    def debug(self, message):
        """Log a debug message."""
        self.get_logger().debug(message)
        
    def info(self, message):
        """Log an info message."""
        self.get_logger().info(message)
        
    def warning(self, message):
        """Log a warning message."""
        self.get_logger().warning(message)
        
    def error(self, message):
        """Log an error message."""
        self.get_logger().error(message)
        
    def critical(self, message):
        """Log a critical message."""
        self.get_logger().critical(message)


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
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
        
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
        return {
            "model": {
                "type": "logistic_regression", 
                "params": {"max_iter": 200}
            },
            "server": {
                "host": "0.0.0.0", 
                "port": 7860, 
                "mcp_server": True
            },
            "client": {
                "server_url": "http://localhost:7860/gradio_api/mcp/sse"
            },
            "logging": {
                "level": "INFO", 
                "file": None
            }
        }
        
    def load_config(self, config_file):
        """
        Load configuration from a file.
        
        Args:
            config_file (str): Path to the configuration file
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
            
        file_ext = os.path.splitext(config_file)[1].lower()
        
        if file_ext == '.json':
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
        elif file_ext in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML config files")
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
            
        # Update config with loaded values
        self._update_nested_dict(self.config, loaded_config)
        
    def _update_nested_dict(self, d, u):
        """Helper method to update nested dictionaries."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        
    def get_config(self, section=None):
        """
        Get the current configuration.
        
        Args:
            section (str, optional): Configuration section to return
            
        Returns:
            dict: Current configuration or section
        """
        if section:
            return self.config.get(section, {})
        return self.config
        
    def update_config(self, updates, section=None):
        """
        Update the configuration with new values.
        
        Args:
            updates (dict): Dictionary of configuration updates
            section (str, optional): Section to update
        """
        if section:
            if section not in self.config:
                self.config[section] = {}
            self._update_nested_dict(self.config[section], updates)
        else:
            self._update_nested_dict(self.config, updates)
        
    def save_config(self, config_file=None):
        """
        Save the current configuration to a file.
        
        Args:
            config_file (str, optional): Path to save the configuration
        """
        save_path = config_file or self.config_file
        if not save_path:
            raise ValueError("No config file path specified")
            
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        file_ext = os.path.splitext(save_path)[1].lower()
        
        if file_ext == '.json':
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        elif file_ext in ['.yaml', '.yml']:
            try:
                import yaml
                with open(save_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML is required to save YAML config files")
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")


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
            tuple: (is_valid, error_message)
                is_valid (bool): True if valid, False otherwise
                error_message (str): Error message if invalid, None if valid
        """
        # Check if features is a list or array-like
        if not isinstance(features, (list, tuple, np.ndarray)):
            return False, f"Features must be a list or array, got {type(features)}"
            
        # Check if features has exactly 4 elements
        if len(features) != 4:
            return False, f"Features must have exactly 4 elements, got {len(features)}"
            
        # Check if all elements can be converted to float
        try:
            features = [float(f) for f in features]
        except (ValueError, TypeError):
            return False, "All feature values must be convertible to float"
            
        # Check if values are within reasonable ranges for Iris dataset
        # These ranges are approximated based on the Iris dataset characteristics
        reasonable_ranges = [
            (4.0, 8.0),  # Sepal length
            (2.0, 4.5),  # Sepal width
            (1.0, 7.0),  # Petal length
            (0.1, 2.5)   # Petal width
        ]
        
        for i, (val, (min_val, max_val)) in enumerate(zip(features, reasonable_ranges)):
            if val < min_val or val > max_val:
                return False, f"Feature {i+1} value {val} is outside reasonable range ({min_val}, {max_val})"
        
        return True, None
        
    @staticmethod
    def validate_model_params(model_type, params):
        """
        Validate model parameters for the specified model type.
        
        Args:
            model_type (str): Type of model
            params (dict): Model parameters
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not isinstance(params, dict):
            return False, "Model parameters must be a dictionary"
            
        # Validate for 'logistic_regression' model type
        if model_type == "logistic_regression":
            valid_params = {
                "max_iter": (int, (10, 10000)),
                "C": (float, (0.001, 1000)),
                "solver": (str, ["lbfgs", "newton-cg", "liblinear", "sag", "saga"]),
                "random_state": (int, None)
            }
            
            for param, value in params.items():
                if param not in valid_params:
                    return False, f"Invalid parameter for logistic_regression: {param}"
                    
                param_type, value_range = valid_params[param]
                
                # Check type
                if not isinstance(value, param_type):
                    return False, f"Parameter {param} should be of type {param_type.__name__}"
                    
                # Check range if specified
                if value_range and isinstance(value_range, list) and value not in value_range:
                    return False, f"Parameter {param} should be one of {value_range}"
                elif value_range and isinstance(value_range, tuple) and (value < value_range[0] or value > value_range[1]):
                    return False, f"Parameter {param} should be in range {value_range}"
                    
        elif model_type not in ["logistic_regression", "decision_tree", "random_forest"]:
            return False, f"Unsupported model type: {model_type}"
            
        return True, None
        
    @staticmethod
    def validate_server_config(config):
        """
        Validate server configuration.
        
        Args:
            config (dict): Server configuration
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not isinstance(config, dict):
            return False, "Server configuration must be a dictionary"
            
        # Check required keys
        required_keys = ["host", "port"]
        for key in required_keys:
            if key not in config:
                return False, f"Missing required configuration key: {key}"
                
        # Validate port
        if not isinstance(config["port"], int):
            return False, "Port must be an integer"
            
        if config["port"] < 1 or config["port"] > 65535:
            return False, f"Port must be in range [1, 65535], got {config['port']}"
            
        # Validate host
        if not isinstance(config["host"], str):
            return False, "Host must be a string"
            
        return True, None


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
        # If features is a string, split by commas and convert to float
        if isinstance(features, str):
            features = [float(x.strip()) for x in features.split(",")]
            
        # If features is a list, convert to numpy array
        if isinstance(features, (list, tuple)):
            features = np.array(features, dtype=float)
            
        # If features is already a numpy array, ensure it's float type
        if isinstance(features, np.ndarray):
            features = features.astype(float)
        else:
            raise TypeError(f"Cannot convert {type(features)} to numpy array")
            
        # Reshape to ensure correct dimensions for scikit-learn
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        return features
        
    @staticmethod
    def dict_to_json(data):
        """
        Convert a dictionary to a JSON string.
        
        Args:
            data (dict): Dictionary to convert
            
        Returns:
            str: JSON string
        """
        try:
            return json.dumps(data, indent=2)
        except (TypeError, OverflowError) as e:
            raise ValueError(f"Error serializing dictionary to JSON: {e}")
        
    @staticmethod
    def json_to_dict(json_str):
        """
        Convert a JSON string to a dictionary.
        
        Args:
            json_str (str): JSON string to convert
            
        Returns:
            dict: Converted dictionary
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON string: {e}")


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
        # Use the current file to determine the project root
        current_file = Path(__file__).resolve()
        # Go up one level from src directory
        return str(current_file.parent.parent)
        
    @staticmethod
    def get_config_path(filename=None):
        """
        Get the path to a configuration file.
        
        Args:
            filename (str, optional): Name of the configuration file
            
        Returns:
            str: Path to the configuration file
        """
        root = PathManager.get_project_root()
        config_dir = os.path.join(root, 'config')
        
        if filename:
            return os.path.join(config_dir, filename)
        return config_dir
        
    @staticmethod
    def get_model_path(filename=None):
        """
        Get the path to a model file.
        
        Args:
            filename (str, optional): Name of the model file
            
        Returns:
            str: Path to the model file
        """
        root = PathManager.get_project_root()
        model_dir = os.path.join(root, 'models')
        
        if filename:
            return os.path.join(model_dir, filename)
        return model_dir
        
    @staticmethod
    def get_log_path(filename=None):
        """
        Get the path to a log file.
        
        Args:
            filename (str, optional): Name of the log file
            
        Returns:
            str: Path to the log file
        """
        root = PathManager.get_project_root()
        log_dir = os.path.join(root, 'logs')
        
        if filename:
            return os.path.join(log_dir, filename)
        return log_dir
        
    @staticmethod
    def ensure_directory_exists(path):
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path (str): Path to the directory
            
        Returns:
            bool: True if directory exists or was created, False otherwise
        """
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            return True
        except Exception as e:
            logging.error(f"Error creating directory {path}: {e}")
            return False
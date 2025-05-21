# Source Code Documentation

This directory contains the source code for the GenAI-Superstream project. The code is organized into modular components with clear separation of concerns.

## Module Overview

### `data.py`

Handles data loading and preprocessing functionality:

- **DataLoader**: Loads the Iris dataset from scikit-learn and provides methods for data splitting
- **DataPreprocessor**: Implements preprocessing strategies for feature data, including scaling and validation

### `model.py`

Implements the machine learning model components:

- **ModelTrainer**: Trains scikit-learn classifiers (LogisticRegression, DecisionTree, RandomForest)
- **Predictor**: Provides prediction functionality using a trained model
- **ModelFactory**: Factory class for creating different types of models

### `server.py`

Implements the Gradio-based MCP server:

- **GradioInterface**: Creates the Gradio interface for human interaction
- **MCPServer**: Configures and launches the MCP server with tool exposure
- **ServerConfig**: Manages server configuration parameters
- **create_mcp_server()**: High-level function to create and launch a server

### `client.py`

Implements the MCP client:

- **MCPClient**: Connects to the MCP server and calls the prediction tool
- **ClientInterface**: Provides a Gradio interface for the client
- **ClientConfig**: Manages client configuration parameters
- **ConnectionManager**: Handles connection to the server with retry logic
- **create_mcp_client()**: High-level function to create and launch a client

### `utils.py`

Provides shared utility functionality:

- **Logger**: Handles application logging
- **ConfigManager**: Manages application configuration (Singleton pattern)
- **InputValidator**: Validates various inputs
- **FormatConverter**: Converts between different data formats
- **PathManager**: Manages file paths across the application

## Dependencies

These modules depend on the following external libraries:

- scikit-learn: For ML model implementation
- gradio: For web UI and MCP server capabilities
- mcp: For MCP client communication
- numpy: For numerical operations

## Module Relationships

- **data.py** → **model.py**: The model is trained using data from the data module
- **model.py** → **server.py**: The server exposes the model's prediction functionality
- **server.py** ← **client.py**: The client communicates with the server
- **utils.py** → all other modules: All modules use the utility functions

## Error Handling

All modules implement comprehensive error handling:

- Input validation to prevent invalid operations
- Descriptive error messages
- Graceful handling of connection errors
- Logging of errors for debugging

## Extension Points

The code is designed for extensibility:

1. **Adding new models**: Add new model types in `ModelFactory.create_model()`
2. **New preprocessing strategies**: Add new strategies in `DataPreprocessor`
3. **Additional MCP tools**: Define new functions and expose them in `MCPServer`
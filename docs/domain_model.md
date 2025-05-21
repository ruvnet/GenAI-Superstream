# GenAI-Superstream Domain Model

This document defines the core entities, relationships, data structures, and behaviors in the GenAI-Superstream system.

## 1. Core Entities

### 1.1. Dataset
Represents the Iris dataset used for training and testing the machine learning model.

**Attributes:**
- `data`: NumPy array of feature values
- `target`: NumPy array of target labels
- `feature_names`: List of feature names (strings)
- `target_names`: List of class names (strings)

**Behaviors:**
- Load data from scikit-learn
- Provide access to features and targets
- Support data splitting (train/test)

### 1.2. Model
Represents the machine learning model used for classification.

**Attributes:**
- `classifier`: The scikit-learn model instance
- `parameters`: Dictionary of model hyperparameters
- `is_trained`: Boolean indicating trained status

**Behaviors:**
- Train on provided data
- Make predictions
- Provide prediction probabilities

### 1.3. Predictor
Encapsulates the prediction functionality exposed as an MCP tool.

**Attributes:**
- `model`: Reference to the trained model
- `dataset`: Reference to dataset for class names

**Behaviors:**
- Accept feature inputs
- Validate inputs
- Return formatted predictions with class probabilities

### 1.4. MCPServer
Represents the server that exposes the model via MCP.

**Attributes:**
- `predictor`: Reference to the Predictor
- `host`: Server hostname (string)
- `port`: Server port (integer)
- `blocks`: Gradio Blocks instance

**Behaviors:**
- Initialize Gradio interface
- Launch MCP server
- Process requests
- Format responses

### 1.5. MCPClient
Represents a client that connects to the MCP server.

**Attributes:**
- `server_url`: URL of the MCP server
- `client`: MCP Client instance

**Behaviors:**
- Connect to server
- Call prediction tool
- Process responses
- Handle errors

## 2. Data Structures

### 2.1. Feature Vector
Representation of the input features for prediction.

**Structure:**
```python
# List of 4 float values
[
    sepal_length,  # float
    sepal_width,   # float
    petal_length,  # float
    petal_width    # float
]
```

### 2.2. Prediction Result
Representation of the model's prediction output.

**Structure:**
```python
# Dictionary mapping class names to probabilities
{
    "setosa": float,      # Probability for setosa class
    "versicolor": float,  # Probability for versicolor class
    "virginica": float    # Probability for virginica class
}
```

### 2.3. MCP Tool Request
Structure of an MCP tool request from client to server.

**Structure:**
```python
{
    "tool": "predict_species",
    "input": {
        "features": [float, float, float, float]
    }
}
```

### 2.4. MCP Tool Response
Structure of an MCP tool response from server to client.

**Structure:**
```python
{
    "result": {
        "setosa": float,
        "versicolor": float,
        "virginica": float
    }
}
```

## 3. Relationships

1. **Dataset → Model**: The Model is trained using data from the Dataset
2. **Model → Predictor**: The Predictor uses the trained Model to make predictions
3. **Predictor → MCPServer**: The MCPServer exposes the Predictor's functionality
4. **MCPClient → MCPServer**: The MCPClient communicates with the MCPServer

## 4. Domain Rules and Constraints

1. Feature values must be floating-point numbers
2. The feature vector must contain exactly 4 values (sepal length, sepal width, petal length, petal width)
3. Probability values must be between 0 and 1
4. The sum of all class probabilities must equal 1
5. The server must properly translate between numeric class indices and human-readable class names
6. Input validation must reject improperly formatted or out-of-range values

## 5. State Transitions

### Model Training:
1. Untrained → Training → Trained

### Server Lifecycle:
1. Initialized → Running → Stopped

### Prediction Process:
1. Input Received → Validated → Processed → Result Returned

## 6. Domain-Specific Terminology

- **Iris Dataset**: A classic machine learning dataset containing measurements of iris flowers
- **Feature**: A measurable property or characteristic used for prediction
- **Class**: A category or label that the model predicts
- **Probability**: A numerical value representing the likelihood of belonging to a specific class
- **MCP (Model Context Protocol)**: A protocol for tool-calling between AI agents and services
- **Gradio**: A Python library for creating customizable web interfaces for ML models
- **LogisticRegression**: A classification algorithm for predicting categorical outcomes

## 7. Domain Events

- **ModelTrained**: Indicates successful completion of model training
- **PredictionRequested**: Triggered when a prediction request is received
- **PredictionCompleted**: Triggered when a prediction is successfully generated
- **ServerStarted**: Indicates the MCP server has successfully started
- **ClientConnected**: Indicates a client has connected to the server
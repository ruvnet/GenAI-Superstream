# GenAI-Superstream Usage Guide

This document provides detailed instructions for using the GenAI-Superstream project, which implements a machine learning model exposed through a Model Context Protocol (MCP) server.

## Project Structure

```
GenAI-Superstream/
├── src/                     # Core implementation
│   ├── data.py              # Data loading and preprocessing
│   ├── model.py             # ML model implementation
│   ├── server.py            # MCP server implementation
│   ├── client.py            # MCP client implementation
│   ├── utils.py             # Utility functions
├── scripts/                 # Executable scripts
│   ├── train_model.py       # Script to train and save models
│   ├── run_server.py        # Script to launch MCP server
│   ├── run_client.py        # Script to launch MCP client
├── main.py                  # Main entry point
├── requirements.txt         # Project dependencies
└── USAGE.md                 # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/GenAI-Superstream.git
   cd GenAI-Superstream
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

The simplest way to run the project is using the `main.py` script with the `simple` command, which implements the approach shown in the README.md:

```bash
python main.py simple
```

This will:
1. Load the Iris dataset
2. Train a LogisticRegression model
3. Launch a Gradio MCP server on localhost:7860

You can then access:
- The web UI at http://localhost:7860
- The MCP endpoint at http://localhost:7860/gradio_api/mcp/sse

## Using the Modular Implementation

For more flexibility and control, you can use the modular implementation:

### 1. Train a Model

```bash
python main.py train --model-type logistic_regression --output models/iris_model.pkl
```

Options:
- `--model-type`: Type of model to train (`logistic_regression`, `decision_tree`, or `random_forest`)
- `--scaling`: Data scaling strategy (`standard` or `minmax`)
- `--output`: Path to save the trained model
- `--config`: Path to a configuration file

Alternatively, use the training script directly:

```bash
python scripts/train_model.py --model-type logistic_regression
```

### 2. Launch the MCP Server

```bash
python main.py server --model models/iris_model.pkl --port 7860
```

Options:
- `--model`: Path to the saved model (if not provided, will train a new one)
- `--host`: Host address to bind the server (default: 0.0.0.0)
- `--port`: Port number (default: 7860)
- `--share`: Create a public share link
- `--config`: Path to a server configuration file

Alternatively, use the server script directly:

```bash
python scripts/run_server.py --model models/iris_model.pkl
```

### 3. Launch the MCP Client

```bash
python main.py client --server http://localhost:7860/gradio_api/mcp/sse --port 7861
```

Options:
- `--server`: URL of the MCP server (default: http://localhost:7860/gradio_api/mcp/sse)
- `--port`: Port for the client interface (default: 7861)
- `--share`: Create a public share link
- `--config`: Path to a client configuration file

Alternatively, use the client script directly:

```bash
python scripts/run_client.py --server http://localhost:7860/gradio_api/mcp/sse
```

## Configuration Files

You can use configuration files to customize the behavior of the components:

### Model Configuration

```yaml
model:
  type: logistic_regression
  params:
    max_iter: 200
    C: 1.0
    random_state: 42
```

### Server Configuration

```yaml
server:
  host: 0.0.0.0
  port: 7860
  share: false
  mcp_server: true
```

### Client Configuration

```yaml
client:
  server_url: http://localhost:7860/gradio_api/mcp/sse
  timeout: 30
  retry_attempts: 3
  retry_delay: 2
```

## Advanced Usage

### Using as a Library

You can also use the components as a library in your own code:

```python
from src.data import DataLoader
from src.model import ModelTrainer, Predictor
from src.server import MCPServer

# Load data
data_loader = DataLoader()
data, target, feature_names, target_names = data_loader.load_iris_dataset()

# Train model
trainer = ModelTrainer(model_type="logistic_regression")
model = trainer.train(data, target)

# Create predictor and server
predictor = Predictor(model, target_names)
server = MCPServer(predictor, host="0.0.0.0", port=7860)

# Launch server
server.launch(share=False)
```

### Testing MCP with curl

You can test the MCP endpoint using curl:

```bash
curl -X POST http://localhost:7860/api/predict \
   -H "Content-Type: application/json" \
   -d '{"data": [5.1, 3.5, 1.4, 0.2]}'
```

The response should contain predictions for the Iris species.

## Troubleshooting

1. **Connection Issues**: Ensure the server is running and accessible from the client.

2. **Model Errors**: Check that the model file exists and is valid.

3. **Input Validation**: Ensure input features are provided as a list of 4 float values.

4. **Dependency Issues**: Verify all required packages are installed with the correct versions.

5. **Port Conflicts**: If the port is already in use, specify a different port using the `--port` option.

## Extending the Project

To extend the project with additional models or capabilities:

1. Add new model types in `src/model.py`
2. Implement new prediction functions in the `Predictor` class
3. Expose these functions via the MCP server in `src/server.py`
4. Update the client to support calling the new functions
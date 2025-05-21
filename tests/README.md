# GenAI-Superstream Test Suite

This directory contains the test suite for the GenAI-Superstream project, including unit tests and integration tests for all components.

## Test Structure

- `test_data.py` - Tests for the data loading and processing functionality
- `test_model.py` - Tests for model training and prediction
- `test_server.py` - Tests for the Gradio server and prediction endpoints
- `test_mcp_integration.py` - Tests specifically for MCP client-server integration

## Running Tests

### Running All Tests

To run all tests, simply use:

```bash
python -m tests.run_tests
```

### Running MCP-Specific Tests

To run only the MCP-related tests (which are most relevant for troubleshooting the "NoneType object is not callable" error):

```bash
python -m tests.run_tests --mcp-only
```

### Running a Specific Test Module

To run all tests in a specific module:

```bash
python -m tests.run_tests --test test_mcp_integration
```

### Running a Specific Test Case

To run all tests in a specific test case:

```bash
python -m tests.run_tests --test test_mcp_integration.TestMCPIntegration
```

### Running a Specific Test Method

To run a specific test method:

```bash
python -m tests.run_tests --test test_mcp_integration.TestMCPIntegration.test_mcp_client_server_interaction
```

### Additional Options

- `--verbose` or `-v`: Enable verbose output
- `--debug`: Enable debug level logging
- `--pattern`: Specify a custom pattern for test discovery

## MCP Debugging

For advanced debugging of MCP client-server interactions, use the debug script:

```bash
python -m scripts.debug_mcp
```

This script will:

1. Start a server with a properly initialized MCP tool
2. Verify the MCP endpoint is accessible
3. Connect with an MCP client
4. Call the prediction tool
5. Report any errors in detail with extensive logging

### Options for the Debug Script

- `--port PORT`: Specify a custom port for testing (default: 7990)
- `--subprocess`: Run the server in a separate process for complete isolation

## Common Issues

### "NoneType object is not callable" Error

This error typically occurs when the MCP client attempts to call a route handler that is not properly initialized. The main causes could be:

1. The predictor is not properly attached to the `predict_species` function
2. The Gradio Blocks instance is not properly created or launched
3. The MCP route handler is not properly registered

The tests in `test_mcp_integration.py` and the debug script are specifically designed to isolate and identify the source of this error.
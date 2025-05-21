#!/bin/bash
# Helper script for the GenAI-Superstream project

# Set up help function
show_help() {
    echo "GenAI-Superstream Helper Script"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup          - Set up the project (create directories, install dependencies)"
    echo "  train          - Train and save the model"
    echo "  server         - Run the MCP server"
    echo "  client         - Run the MCP client"
    echo "  simple         - Run the simplified version (like README example)"
    echo "  test           - Run tests"
    echo "  help           - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh setup"
    echo "  ./run.sh train --model-type logistic_regression"
    echo "  ./run.sh server --port 7860"
    echo "  ./run.sh client --server http://localhost:7860/gradio_api/mcp/sse"
    echo ""
}

# Make sure the script fails if any command fails
set -e

# Extract the command
COMMAND=$1
shift

# Handle no command
if [ -z "$COMMAND" ]; then
    show_help
    exit 0
fi

# Process commands
case $COMMAND in
    setup)
        echo "Setting up GenAI-Superstream..."
        # Create necessary directories
        mkdir -p models logs
        # Install dependencies
        pip install -r requirements.txt
        # Install the package in development mode
        pip install -e .
        echo "Setup complete!"
        ;;
    train)
        echo "Training model..."
        python main.py train "$@"
        ;;
    server)
        echo "Starting MCP server..."
        python main.py server "$@"
        ;;
    client)
        echo "Starting MCP client..."
        python main.py client "$@"
        ;;
    simple)
        echo "Running simplified version..."
        python main.py simple "$@"
        ;;
    test)
        echo "Running tests..."
        python tests/run_tests.py
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
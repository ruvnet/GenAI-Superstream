"""
Main MCP server implementation.
This file initializes the FastMCP server and imports all tools, resources, and prompts.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP

# Import config management
from .config import load_config


@dataclass
class AppContext:
    """
    Type-safe application context container.
    Store any application-wide state or connections here.
    """
    config: dict


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Application lifecycle manager.
    Handles startup and shutdown operations with proper resource management.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        The application context with initialized resources
    """
    # Load configuration
    config = load_config()
    
    # Initialize connections and resources
    print("🚀 Server starting up...")
    
    try:
        # Create and yield the app context
        yield AppContext(config=config)
    finally:
        # Clean up resources on shutdown
        print("🛑 Server shutting down...")


# Create the MCP server with lifespan support
mcp = FastMCP(
    "genai-mcp",  # Server name
    lifespan=app_lifespan,           # Lifecycle manager
    dependencies=["mcp>=1.0"],       # Required dependencies
)

# Import all tools, resources, and prompts
# These imports must come after the MCP server is initialized
from .tools.sample_tools import *
from .resources.sample_resources import *
from .prompts.sample_prompts import *

# Make the server instance accessible to other modules
server = mcp

if __name__ == "__main__":
    # When executed directly, run the server in HTTP/SSE mode
    import os
    import sys

    debug = os.environ.get("MCP_DEBUG", "false").lower() in ("true", "1", "yes")

    if debug:
        print(f"Debug mode: enabled", file=sys.stderr)
        print(f"Starting FastMCP in HTTP/SSE mode", file=sys.stderr)

    try:
        # Use only the transport parameter
        mcp.run(transport="sse")
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        if debug:
            import traceback
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)
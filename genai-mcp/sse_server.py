"""
HTTP/SSE Server implementation for genai-mcp.

This file provides a FastAPI-based HTTP server with Server-Sent Events (SSE)
support for the genai-mcp MCP server.
"""

import os
import json
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn
import logging

# Import the tools, resources, and prompts from genai_mcp
from genai_mcp.tools.sample_tools import echo, calculate, long_task, fetch_data
from genai_mcp.resources.sample_resources import static_resource, dynamic_resource, config_resource, file_resource
from genai_mcp.prompts.sample_prompts import simple_prompt, structured_prompt, data_analysis_prompt, image_analysis_prompt
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("genai-mcp")

# Create FastAPI app
app = FastAPI(
    title="GenAI-MCP Server",
    description="MCP server with HTTP/SSE support",
    version="0.0.1"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"]   # Allow all headers
)

# Register all available tools
tools = {
    "echo": echo,
    "calculate": calculate,
    "long_task": long_task,
    "fetch_data": fetch_data
}

# Register all available resources
resources = {
    "static://example": static_resource,
    "dynamic://{parameter}": dynamic_resource,
    "config://{section}": config_resource,
    "file://{path}.md": file_resource
}

# Register all available prompts
prompts = {
    "simple_prompt": simple_prompt,
    "structured_prompt": structured_prompt,
    "data_analysis_prompt": data_analysis_prompt,
    "image_analysis_prompt": image_analysis_prompt
}

# Mock implementation of run_mcp_request
async def run_mcp_request(payload: Dict[str, Any], tools: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an MCP request and return the result.
    
    Args:
        payload: The MCP request payload
        tools: Dictionary of available tools
        
    Returns:
        The MCP response
    """
    try:
        if "name" not in payload:
            return {"error": "Missing 'name' field in request"}
            
        tool_name = payload.get("name")
        arguments = payload.get("arguments", {})
        
        if tool_name not in tools:
            return {"error": f"Unknown tool: {tool_name}"}
            
        tool_fn = tools[tool_name]
        result = tool_fn(**arguments)
        
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# Define the MCP endpoints
@app.get("/mcp")
async def mcp_get_endpoint(request: Request):
    """
    MCP GET endpoint for SSE connection establishment.
    
    Args:
        request: The HTTP request
        
    Returns:
        An SSE response with a connection success message
    """
    async def event_stream():
        # Send a connection established message
        yield {"event": "connected", "data": json.dumps({"status": "connected"})}
        
    return EventSourceResponse(event_stream())

@app.post("/mcp")
async def mcp_post_endpoint(request: Request):
    """
    MCP POST endpoint that handles tool invocation.
    
    Args:
        request: The HTTP request
        
    Returns:
        An SSE response with the tool result
    """
    try:
        payload = await request.json()
        result = await run_mcp_request(payload, tools)
        
        async def event_stream():
            yield {"event": "result", "data": json.dumps(result)}
            
        return EventSourceResponse(event_stream())
    except Exception as e:
        async def error_stream():
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
            
        return EventSourceResponse(error_stream())

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status and available tools
    """
    return {
        "status": "healthy",
        "tools": list(tools.keys()),
        "resources": list(resources.keys()),
        "prompts": list(prompts.keys())
    }

# Run the server
if __name__ == "__main__":
    debug = os.environ.get("MCP_DEBUG", "false").lower() in ("true", "1", "yes")
    port = int(os.environ.get("MCP_PORT", "8001"))  # Use port 8001 to avoid conflicts
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    
    print(f"🚀 Starting genai-mcp HTTP/SSE server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

import gradio as gr
from mcp import Client

# Connect to the MCP server
client = Client("http://localhost:7880/gradio_api/mcp/sse")

def call_hello_world(name: str):
    """
    Calls the hello_world function on the MCP server
    
    Args:
        name: The name to send to the hello_world function
        
    Returns:
        The greeting response from the server
    """
    # Call the MCP tool named "hello_world"
    response = client.call(
        tool="hello_world",
        input={"name": name}
    )
    return response

# Create a simple Gradio interface
iface = gr.Interface(
    fn=call_hello_world,
    inputs=gr.Textbox(label="Your Name", value="World"),
    outputs=gr.Textbox(label="Server Response"),
    title="Hello World MCP Client",
    description="This client connects to the Hello World MCP server and calls the hello_world function."
)

if __name__ == "__main__":
    iface.launch()
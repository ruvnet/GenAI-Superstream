# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . /app/

# Install uv package manager and dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --no-cache-dir -e ".[dev]"

# Expose the port the server runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the server
CMD ["python", "-m", "genai_mcp.server"]
# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only src and config directories
COPY src/ /app/src/
COPY config/ /app/config/

# Copy requirements.txt for dependencies
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (default FastAPI port)
EXPOSE 8000

# Set PYTHONPATH to include the app directory
ENV PYTHONPATH=/app

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "src.controllers.main_controller:app", "--host", "0.0.0.0", "--port", "8000"]

# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set environment variables to avoid interactive prompts during installation
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=off

# Install system dependencies for building packages and running the script
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6-dev \
    libpng-dev \
    libzmq3-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory for your application
WORKDIR /app

# Copy your Python requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

CMD ["python", "test.py"]

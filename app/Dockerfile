# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set environment variables 
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements file
COPY requirements.txt .

# Install the large package separately with increased timeout
RUN pip install --no-cache-dir --default-timeout=1000 torch

# Install other Python dependencies with increased timeout
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on 
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]


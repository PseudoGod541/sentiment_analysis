# Use a Python version compatible with your TensorFlow model
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first for better caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your application files into the container
COPY . .

# The specific command to run will be provided by docker-compose.yml

# Use a lightweight version of Python
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install necessary libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code into the container
COPY code/ ./code/
COPY data/ ./data/
COPY models/ ./models/

# Command to run the code
ENTRYPOINT ["python", "code/imputation.py"]
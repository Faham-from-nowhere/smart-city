# Use an official Python runtime as a parent image
# Changed from 3.10 to 3.11 to match ipython requirement (and potentially other packages)
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies, including git and ca-certificates
# 'ca-certificates' is crucial for establishing secure (HTTPS) connections
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code and DATA files into the container
COPY api_service.py .
COPY traffic_prediction_model.pkl .
COPY processed_combined_data_hyd.csv . # <--- ADD THIS LINE!

# If your FastAPI service needs to read latest_pollution.json at startup (e.g., for fallbacks),
# ensure that's copied too if it's directly referenced in startup_event
# COPY latest_pollution.json .

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]
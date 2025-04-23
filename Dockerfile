# # Use Python 3.9 as base image
# FROM python:3.9-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements first to leverage Docker cache
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application
# COPY . .

# # Create necessary directories
# RUN mkdir -p logs mlruns

# # Set environment variables
# ENV PYTHONUNBUFFERED=1
# ENV STREAMLIT_SERVER_PORT=8501
# ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# # Expose the port
# EXPOSE 8501
# # EXPOSE 8501

# # Command to run the application
# CMD ["streamlit", "run", "app.py"]
# # CMD ["streamlit", "run", "app.py", "&&", "mlflow", "server", "--backend-store-uri", "sqlite:///mlruns/mlflow.db"]
# ENTRYPOINT ["mlflow", "server", "--backend-store-uri", "sqlite:///mlruns/mlflow.db"]
# EXPOSE 5000


# # CMD ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlruns/mlflow.db"]

# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p logs mlruns

# Copy and make start script executable
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV YOUTUBE_API_KEY=api_key

# Expose ports for Streamlit and MLflow
EXPOSE 8501
EXPOSE 5000

# Start both MLflow and Streamlit using the shell script
CMD ["/app/start.sh"]


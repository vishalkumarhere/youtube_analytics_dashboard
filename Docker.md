# Docker Setup Guide for YouTube Analytics Dashboard

## Table of Contents
1. [Project Structure](#project-structure)
2. [Prerequisites](#prerequisites)
3. [Docker Configuration](#docker-configuration)
4. [Environment Setup](#environment-setup)
5. [Building and Running](#building-and-running)
6. [Volumes and Persistence](#volumes-and-persistence)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Project Structure
```
.
├── app.py                 # Main Streamlit application
├── youtube_api.py         # YouTube API integration
├── model_monitoring_utils.py  # MLflow monitoring utilities
├── logging_config.py      # Logging configuration
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image configuration
├── docker-compose.yml    # Docker Compose configuration
├── .env                  # Environment variables
├── logs/                 # Application logs directory
└── mlruns/              # MLflow tracking directory
```

## Prerequisites
- Docker Engine (version 20.10.0 or higher)
- Docker Compose (version 2.0.0 or higher)
- A valid YouTube API key
- At least 2GB of available memory
- Port 8501 available for Streamlit

## Docker Configuration

### Dockerfile
The project uses a multi-stage build process to optimize the final image size:

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better cache utilization
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p logs mlruns

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py"]
```

### Docker Compose Configuration
The `docker-compose.yml` file orchestrates the application services:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./logs:/app/logs
      - ./mlruns:/app/mlruns
    env_file:
      - .env
    environment:
      - MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
    restart: unless-stopped
```

## Environment Setup

1. Create a `.env` file in the project root:
```env
YOUTUBE_API_KEY=your_api_key_here
STREAMLIT_THEME_BASE=light
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

2. Required environment variables:
- `YOUTUBE_API_KEY`: Your YouTube Data API v3 key
- `MLFLOW_TRACKING_URI`: URI for MLflow tracking (set in docker-compose.yml)
- `STREAMLIT_THEME_BASE`: UI theme configuration
- `PYTHONUNBUFFERED`: Ensures Python output is sent straight to container logs

## Building and Running

### Development Build
```bash
# Build the image
docker-compose build

# Start the services
docker-compose up

# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f
```

### Production Build
```bash
# Build with production optimizations
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Start production services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Stopping the Application
```bash
# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Volumes and Persistence

The application uses two Docker volumes for data persistence:

1. **Logs Volume**
   - Mount: `./logs:/app/logs`
   - Purpose: Stores application logs
   - Persistence: Survives container restarts
   - Backup: Regular backup recommended

2. **MLflow Volume**
   - Mount: `./mlruns:/app/mlruns`
   - Purpose: Stores ML model tracking data
   - Persistence: Survives container restarts
   - Contains: Model metrics, parameters, and artifacts

## Troubleshooting

### Common Issues and Solutions

1. **Container fails to start**
   ```bash
   # Check container logs
   docker-compose logs app
   
   # Verify environment variables
   docker-compose config
   ```

2. **Permission issues with volumes**
   ```bash
   # Fix permissions on host
   sudo chown -R 1000:1000 logs/ mlruns/
   ```

3. **Memory issues**
   ```bash
   # Check container resource usage
   docker stats
   
   # Increase container memory limit in docker-compose.yml
   services:
     app:
       deploy:
         resources:
           limits:
             memory: 4G
   ```

4. **Port conflicts**
   ```bash
   # Check port usage
   sudo lsof -i :8501
   
   # Change port mapping in docker-compose.yml
   ports:
     - "8502:8501"
   ```

## Best Practices

1. **Security**
   - Never commit `.env` file
   - Use secrets management for sensitive data
   - Regularly update base images
   - Implement least privilege principle

2. **Performance**
   - Use multi-stage builds
   - Optimize layer caching
   - Minimize image size
   - Use appropriate base images

3. **Monitoring**
   - Implement health checks
   - Monitor container resources
   - Set up log rotation
   - Use container orchestration for production

4. **Development Workflow**
   ```bash
   # Create a development branch
   git checkout -b feature/new-feature

   # Build development version
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

   # Run tests
   docker-compose exec app python -m pytest

   # Check logs
   docker-compose logs -f app
   ```

5. **Backup Strategy**
   ```bash
   # Backup volumes
   docker run --rm \
     -v youtube_analytics_logs:/logs \
     -v $(pwd)/backup:/backup \
     alpine tar czf /backup/logs_backup.tar.gz /logs

   # Backup MLflow data
   docker run --rm \
     -v youtube_analytics_mlruns:/mlruns \
     -v $(pwd)/backup:/backup \
     alpine tar czf /backup/mlruns_backup.tar.gz /mlruns
   ```

## Additional Notes

1. **Resource Requirements**
   - CPU: Minimum 2 cores recommended
   - RAM: Minimum 4GB recommended
   - Storage: At least 10GB free space

2. **Scaling Considerations**
   - Implement load balancing for multiple instances
   - Use container orchestration (Kubernetes/Swarm)
   - Consider using managed services for MLflow

3. **Maintenance**
   - Regular security updates
   - Log rotation and cleanup
   - Monitor disk usage
   - Backup verification

4. **Network Configuration**
   - Internal ports: 8501 (Streamlit)
   - External ports: Configurable in docker-compose.yml
   - Network mode: bridge (default)

5. **Production Deployment**
   - Use production-grade web server
   - Implement SSL/TLS
   - Set up monitoring and alerting
   - Configure automatic restarts 
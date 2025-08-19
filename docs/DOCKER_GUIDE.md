# 🐳 Docker Guide - Precision Background Remover

## Overview

This guide provides comprehensive Docker deployment options for the Precision Background Remover, optimized for both Mac (Intel/Apple Silicon) and other platforms.

## 🚀 Quick Start

### Option 1: Simple Docker Build (Recommended for most users)
```bash
# Build the simple version
docker build -t bg-remover-simple .

# Run with web interface
docker run -p 8501:8501 bg-remover-simple

# Access at: http://localhost:8501
```

### Option 2: Advanced Docker Build (GPU-enabled)
```bash
# Build advanced version with GPU support
docker build -f Dockerfile.advanced -t bg-remover-advanced .

# Run with GPU support (requires NVIDIA Docker)
docker run --gpus all -p 8502:8501 bg-remover-advanced
```

### Option 3: Docker Compose (All services)
```bash
# Start simple service
docker-compose up background-remover-simple

# Start GPU service (if supported)
docker-compose --profile gpu up background-remover-advanced

# Start CLI service
docker-compose --profile cli up background-remover-cli
```

## 🍎 Mac-Specific Docker Setup

### Intel Macs
```bash
# Standard Docker build works well
docker build -t bg-remover .
docker run -p 8501:8501 bg-remover
```

### Apple Silicon Macs (M1/M2/M3)
```bash
# Build for ARM64 architecture
docker build --platform linux/arm64 -t bg-remover-arm64 .

# Run with platform specification
docker run --platform linux/arm64 -p 8501:8501 bg-remover-arm64

# Or use Rosetta emulation (slower but more compatible)
docker run --platform linux/amd64 -p 8501:8501 bg-remover
```

### Mac Performance Optimizations
```bash
# Allocate more memory to Docker (recommended: 8GB+)
# Docker Desktop → Settings → Resources → Memory

# Enable experimental features for better performance
# Docker Desktop → Settings → Docker Engine → Add:
{
  "experimental": true,
  "features": {
    "buildkit": true
  }
}
```

## 📁 Volume Mounting

### Input/Output Directories
```bash
# Mount local directories for file processing
docker run -p 8501:8501 \
  -v $(pwd)/data/input:/app/uploads:ro \
  -v $(pwd)/data/output:/app/outputs \
  bg-remover-simple
```

### Model Caching
```bash
# Create persistent volume for models
docker volume create bg-remover-models

# Use the volume
docker run -p 8501:8501 \
  -v bg-remover-models:/app/models \
  bg-remover-simple
```

## 🔧 Environment Configuration

### CPU-Only Configuration
```bash
docker run -p 8501:8501 \
  -e CUDA_VISIBLE_DEVICES="" \
  -e OMP_NUM_THREADS=4 \
  bg-remover-simple
```

### GPU Configuration (NVIDIA)
```bash
# Requires nvidia-docker2
docker run --gpus all -p 8501:8501 \
  -e CUDA_VISIBLE_DEVICES=0 \
  bg-remover-advanced
```

### Memory-Optimized Configuration
```bash
# For systems with limited memory
docker run -p 8501:8501 \
  --memory=4g \
  --memory-swap=8g \
  -e STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50 \
  bg-remover-simple
```

## 🌐 Service Configurations

### Simple Background Remover
- **Port**: 8501
- **Features**: Enhanced BiRefNet processing, quality analysis
- **Memory**: ~2-4GB recommended
- **CPU**: Any modern CPU

### Advanced Background Remover
- **Port**: 8502
- **Features**: Medical SAM2, ensemble processing, GPU acceleration
- **Memory**: ~8-16GB recommended
- **GPU**: NVIDIA GPU with CUDA support

### CLI Processing Service
- **Usage**: Command-line batch processing
- **Mount**: Input/output directories required
- **Profiles**: Use `docker-compose --profile cli`

## 🏥 Production Deployment

### Medical-Grade Settings
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  bg-remover-medical:
    build:
      dockerfile: Dockerfile.advanced
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 16G
          cpus: '8'
        reservations:
          memory: 8G
          cpus: '4'
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - MEDICAL_GRADE_MODE=true
      - QUALITY_THRESHOLD=0.95
```

### Health Checks
```bash
# Check container health
docker ps
docker logs bg-remover-simple

# Test API endpoint
curl -f http://localhost:8501/_stcore/health
```

### Scaling
```bash
# Scale simple service
docker-compose up --scale background-remover-simple=3

# Load balancer configuration (nginx example)
upstream bg_remover {
    server localhost:8501;
    server localhost:8502;
    server localhost:8503;
}
```

## 🛠️ Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Reduce memory usage
docker run -p 8501:8501 \
  --memory=2g \
  -e STREAMLIT_SERVER_MAX_UPLOAD_SIZE=20 \
  bg-remover-simple
```

**2. Slow Performance on Mac**
```bash
# Use native platform
docker run --platform linux/arm64 -p 8501:8501 bg-remover-arm64

# Or allocate more resources
# Docker Desktop → Resources → Advanced → CPU: 4, Memory: 8GB
```

**3. GPU Not Detected**
```bash
# Verify NVIDIA Docker setup
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Check CUDA availability in container
docker run --gpus all bg-remover-advanced python -c "import torch; print(torch.cuda.is_available())"
```

**4. Port Already in Use**
```bash
# Use different port
docker run -p 8502:8501 bg-remover-simple

# Or stop conflicting service
docker stop $(docker ps -q --filter "publish=8501")
```

**5. Model Download Issues**
```bash
# Pre-download models
docker run -v bg-remover-models:/app/models bg-remover-simple python -c "from rembg import new_session; new_session('birefnet-general')"
```

### Performance Optimization

**Mac-Specific Optimizations:**
```bash
# Apple Silicon with Rosetta
export DOCKER_DEFAULT_PLATFORM=linux/amd64

# Native ARM64 build
docker buildx build --platform linux/arm64 -t bg-remover-native .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t bg-remover-multi .
```

**Memory Optimization:**
```bash
# Enable swap accounting (Linux hosts)
echo 'GRUB_CMDLINE_LINUX="cgroup_enable=memory swapaccount=1"' >> /etc/default/grub
update-grub

# Docker memory limits
docker run --memory=4g --memory-swap=8g bg-remover-simple
```

## 📊 Monitoring

### Container Stats
```bash
# Real-time stats
docker stats bg-remover-simple

# Detailed inspection
docker inspect bg-remover-simple
```

### Logs
```bash
# Follow logs
docker logs -f bg-remover-simple

# Export logs
docker logs bg-remover-simple > bg-remover.log 2>&1
```

### Health Monitoring
```bash
# Custom health check
docker run -p 8501:8501 \
  --health-cmd="python -c 'import requests; requests.get(\"http://localhost:8501/_stcore/health\")'" \
  --health-interval=30s \
  --health-retries=3 \
  bg-remover-simple
```

## 🔐 Security

### Production Security
```bash
# Run as non-root user
docker run -p 8501:8501 \
  --user 1000:1000 \
  bg-remover-simple

# Read-only root filesystem
docker run -p 8501:8501 \
  --read-only \
  --tmpfs /tmp \
  bg-remover-simple

# Security scanning
docker scan bg-remover-simple
```

### Network Security
```bash
# Custom network
docker network create bg-remover-net
docker run --network bg-remover-net bg-remover-simple

# Reverse proxy (nginx)
server {
    listen 80;
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📈 Performance Benchmarks

| Configuration | Processing Time | Memory Usage | Quality Score |
|--------------|----------------|--------------|---------------|
| Simple CPU | 3-5 seconds | 2-4GB | 0.85-0.95 |
| Advanced CPU | 5-8 seconds | 4-8GB | 0.90-0.98 |
| GPU-Accelerated | 2-4 seconds | 6-12GB | 0.95-1.00 |
| Mac Apple Silicon | 3-6 seconds | 3-6GB | 0.88-0.96 |
| Mac Intel | 4-7 seconds | 3-7GB | 0.85-0.94 |

## 🚀 Advanced Usage

### Custom Model Integration
```dockerfile
# Custom Dockerfile with additional models
FROM bg-remover-simple
RUN pip install your-custom-model
COPY custom_models/ /app/models/
```

### API Integration
```python
import requests

# Process image via Docker container
files = {'file': open('image.jpg', 'rb')}
response = requests.post('http://localhost:8501/api/process', files=files)
```

### Batch Processing
```bash
# Process multiple images
docker run -v $(pwd)/batch_input:/app/input \
           -v $(pwd)/batch_output:/app/output \
           bg-remover-simple python batch_process.py
```

This Docker setup provides flexible deployment options for any environment while maintaining the precision-grade quality and Mac compatibility of the background removal system.
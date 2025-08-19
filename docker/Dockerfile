# Medical-Grade Background Remover - Docker Image
# Optimized for both x86_64 and ARM64 (Apple Silicon) architectures

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables for optimal performance
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    cmake \
    pkg-config \
    # Image processing libraries
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Additional libraries for enhanced processing
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    # Networking and utilities
    wget \
    curl \
    git \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements_simple.txt requirements.txt ./

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    # Install NumPy first with specific version for compatibility
    pip install --no-cache-dir "numpy<2.0.0" && \
    # Install simple requirements (stable)
    pip install --no-cache-dir -r requirements_simple.txt && \
    # Try to install advanced requirements (graceful failure)
    (pip install --no-cache-dir -r requirements.txt || echo "Advanced requirements skipped - using stable version") && \
    # Clear pip cache
    pip cache purge

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/outputs /app/models

# Set proper permissions
RUN chmod +x demo_working.py && \
    chmod -R 755 /app/src

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import cv2, numpy as np; print('Health check passed')" || exit 1

# Expose Streamlit port
EXPOSE 8501

# Default command - run the simple web interface
CMD ["streamlit", "run", "app_simple.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
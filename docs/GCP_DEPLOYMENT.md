# Google Cloud Platform (GCP) Deployment Guide

This guide provides comprehensive instructions for deploying the Precision Background Remover on Google Cloud Platform, including various deployment options and best practices.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Cloud Run Deployment](#cloud-run-deployment)
3. [Google Kubernetes Engine (GKE)](#google-kubernetes-engine-gke)
4. [Compute Engine Virtual Machines](#compute-engine-virtual-machines)
5. [Vertex AI Custom Containers](#vertex-ai-custom-containers)
6. [App Engine Flexible](#app-engine-flexible)
7. [Cloud Storage Integration](#cloud-storage-integration)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Cost Optimization](#cost-optimization)
10. [Security Considerations](#security-considerations)

## Prerequisites

### Required Services
- Google Cloud Platform account with billing enabled
- Google Cloud CLI (`gcloud`) installed and configured
- Docker installed locally
- Project with necessary APIs enabled

### Enable Required APIs
```bash
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    container.googleapis.com \
    compute.googleapis.com \
    storage.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com \
    artifactregistry.googleapis.com
```

### Set Environment Variables
```bash
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export SERVICE_NAME="precision-bg-remover"
export IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
```

## Cloud Run Deployment

Cloud Run is ideal for serverless, containerized applications with automatic scaling.

### 1. Create Optimized Dockerfile for Cloud Run

Create `Dockerfile.gcp`:

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PORT=8080
ENV PYTHONPATH=/app

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### 2. Build and Deploy to Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --region ${REGION} \
    --platform managed \
    --memory 4Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 10 \
    --concurrency 1 \
    --allow-unauthenticated \
    --set-env-vars="PRECISION_MODE=ultra_high" \
    --set-env-vars="ENABLE_GPU=false"
```

### 3. Custom Domain Setup

```bash
# Map custom domain
gcloud run domain-mappings create \
    --service ${SERVICE_NAME} \
    --domain yourdomain.com \
    --region ${REGION}
```

## Google Kubernetes Engine (GKE)

For production workloads requiring more control and GPU support.

### 1. Create GKE Cluster with GPU Nodes

```bash
# Create cluster
gcloud container clusters create precision-bg-cluster \
    --region ${REGION} \
    --machine-type n1-standard-4 \
    --num-nodes 2 \
    --enable-autoscaling \
    --min-nodes 1 \
    --max-nodes 5 \
    --enable-autorepair \
    --enable-autoupgrade

# Add GPU node pool
gcloud container node-pools create gpu-pool \
    --cluster precision-bg-cluster \
    --region ${REGION} \
    --machine-type n1-standard-4 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --num-nodes 0 \
    --enable-autoscaling \
    --min-nodes 0 \
    --max-nodes 3 \
    --node-taints nvidia.com/gpu=present:NoSchedule
```

### 2. Install NVIDIA GPU Device Plugin

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### 3. Deploy Application to GKE

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: precision-bg-remover
spec:
  replicas: 2
  selector:
    matchLabels:
      app: precision-bg-remover
  template:
    metadata:
      labels:
        app: precision-bg-remover
    spec:
      containers:
      - name: app
        image: gcr.io/PROJECT_ID/precision-bg-remover
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        env:
        - name: PRECISION_MODE
          value: "ultra_high"
        - name: ENABLE_GPU
          value: "true"
        livenessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
---
apiVersion: v1
kind: Service
metadata:
  name: precision-bg-service
spec:
  selector:
    app: precision-bg-remover
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

Apply the configuration:

```bash
kubectl apply -f k8s/deployment.yaml
```

## Compute Engine Virtual Machines

For custom VM deployments with full control.

### 1. Create GPU-Enabled VM Instance

```bash
gcloud compute instances create precision-bg-vm \
    --zone us-central1-c \
    --machine-type n1-standard-4 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --image-family ubuntu-2004-lts \
    --image-project ubuntu-os-cloud \
    --boot-disk-size 100GB \
    --boot-disk-type pd-ssd \
    --maintenance-policy TERMINATE \
    --tags http-server,https-server \
    --metadata startup-script='#!/bin/bash
    # Install NVIDIA drivers
    curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
    sudo python3 install_gpu_driver.py
    
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    
    # Install nvidia-docker
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
    '
```

### 2. Deploy Application on VM

SSH into the VM and run:

```bash
# Clone repository
git clone <your-repo-url>
cd Background-Remover

# Build and run with GPU support
docker build -f docker/Dockerfile.gpu -t precision-bg-remover .
docker run -d --gpus all -p 80:8501 --name precision-app precision-bg-remover
```

## Vertex AI Custom Containers

For ML workloads with managed infrastructure.

### 1. Create Custom Container for Vertex AI

Create `Dockerfile.vertex`:

```dockerfile
FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Vertex AI expects specific port
ENV AIP_HTTP_PORT=8080
EXPOSE 8080

CMD ["python", "vertex_ai_service.py"]
```

### 2. Create Vertex AI Service Script

Create `vertex_ai_service.py`:

```python
import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from src.core import remove_background_precision_grade

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        image_data = request.json['instances'][0]['image']
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image
        result, metrics = remove_background_precision_grade(
            image,
            precision_mode='ultra_high'
        )
        
        # Encode result
        _, buffer = cv2.imencode('.png', result)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'predictions': [{
                'image': result_b64,
                'metrics': metrics
            }]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('AIP_HTTP_PORT', 8080))
    app.run(host='0.0.0.0', port=port)
```

### 3. Deploy to Vertex AI

```bash
# Build and push
gcloud builds submit --tag gcr.io/${PROJECT_ID}/vertex-precision-bg

# Deploy to Vertex AI
gcloud ai models upload \
    --region=${REGION} \
    --display-name=precision-bg-remover \
    --container-image-uri=gcr.io/${PROJECT_ID}/vertex-precision-bg \
    --container-ports=8080
```

## App Engine Flexible

For automatic scaling with minimal configuration.

### 1. Create app.yaml

```yaml
runtime: custom
env: flex

automatic_scaling:
  min_num_instances: 1
  max_num_instances: 10
  cool_down_period_sec: 180
  cpu_utilization:
    target_utilization: 0.6

resources:
  cpu: 2
  memory_gb: 4
  disk_size_gb: 20

env_variables:
  PRECISION_MODE: 'ultra_high'
  ENABLE_GPU: 'false'

health_check:
  enable_health_check: true
  check_interval_sec: 30
  timeout_sec: 10
  unhealthy_threshold: 3
  healthy_threshold: 2
```

### 2. Deploy to App Engine

```bash
gcloud app deploy app.yaml --project ${PROJECT_ID}
```

## Cloud Storage Integration

### 1. Create Storage Buckets

```bash
# Create buckets for input and output
gsutil mb -l ${REGION} gs://${PROJECT_ID}-input-images
gsutil mb -l ${REGION} gs://${PROJECT_ID}-output-images

# Set lifecycle policies
cat > lifecycle.json << EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 30}
      }
    ]
  }
}
EOF

gsutil lifecycle set lifecycle.json gs://${PROJECT_ID}-input-images
gsutil lifecycle set lifecycle.json gs://${PROJECT_ID}-output-images
```

### 2. Batch Processing with Cloud Functions

Create `batch_processor.py`:

```python
import functions_framework
from google.cloud import storage
import cv2
import numpy as np
from src.core import remove_background_precision_grade

@functions_framework.cloud_event
def process_uploaded_image(cloud_event):
    """Process image when uploaded to Cloud Storage."""
    
    # Get file info
    file_data = cloud_event.data
    bucket_name = file_data['bucket']
    file_name = file_data['name']
    
    # Download image
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # Process image
    image_bytes = blob.download_as_bytes()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Remove background
    result, metrics = remove_background_precision_grade(image)
    
    # Save result
    _, buffer = cv2.imencode('.png', result)
    
    output_bucket = client.bucket(f"{bucket_name.replace('-input', '-output')}")
    output_blob = output_bucket.blob(f"processed_{file_name}")
    output_blob.upload_from_string(buffer.tobytes(), content_type='image/png')
    
    print(f"Processed {file_name} with quality score: {metrics.get('overall_quality', 0)}")
```

## Monitoring and Logging

### 1. Enable Cloud Monitoring

```bash
# Create custom metrics
gcloud logging metrics create processing_time \
    --description="Average processing time" \
    --log-filter='resource.type="cloud_run_revision" AND "Processing completed"'

gcloud logging metrics create quality_score \
    --description="Quality scores" \
    --log-filter='resource.type="cloud_run_revision" AND "quality_score"'
```

### 2. Set Up Alerting

```bash
# Create alert policy for high processing time
gcloud alpha monitoring policies create \
    --policy-from-file=monitoring/processing_time_alert.yaml
```

Create `monitoring/processing_time_alert.yaml`:

```yaml
displayName: High Processing Time Alert
conditions:
  - displayName: Processing time > 60s
    conditionThreshold:
      filter: resource.type="cloud_run_revision"
      comparison: COMPARISON_GREATER_THAN
      thresholdValue: 60
      duration: 300s
      aggregations:
        - alignmentPeriod: 60s
          perSeriesAligner: ALIGN_MEAN
notificationChannels:
  - projects/PROJECT_ID/notificationChannels/NOTIFICATION_CHANNEL_ID
```

## Cost Optimization

### 1. Use Preemptible Instances

```bash
# Create cost-optimized GKE cluster
gcloud container clusters create precision-bg-cost-optimized \
    --region ${REGION} \
    --preemptible \
    --machine-type n1-standard-2 \
    --num-nodes 2 \
    --enable-autoscaling \
    --min-nodes 0 \
    --max-nodes 5
```

### 2. Implement Smart Scaling

```yaml
# HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: precision-bg-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: precision-bg-remover
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 3. Budget Alerts

```bash
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="Precision BG Remover Budget" \
    --budget-amount=100USD \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=90 \
    --threshold-rule=percent=100
```

## Security Considerations

### 1. IAM and Service Accounts

```bash
# Create service account
gcloud iam service-accounts create precision-bg-sa \
    --display-name="Precision Background Remover Service Account"

# Assign minimal permissions
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:precision-bg-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:precision-bg-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.objectCreator"
```

### 2. VPC and Firewall Rules

```bash
# Create VPC
gcloud compute networks create precision-bg-vpc --subnet-mode regional

gcloud compute networks subnets create precision-bg-subnet \
    --network precision-bg-vpc \
    --range 10.0.0.0/24 \
    --region ${REGION}

# Create firewall rules
gcloud compute firewall-rules create allow-precision-bg \
    --network precision-bg-vpc \
    --allow tcp:80,tcp:443,tcp:8501 \
    --source-ranges 0.0.0.0/0
```

### 3. Private Container Registry

```bash
# Configure Artifact Registry
gcloud artifacts repositories create precision-bg-repo \
    --repository-format=docker \
    --location=${REGION}

# Build and push to private registry
gcloud builds submit \
    --tag ${REGION}-docker.pkg.dev/${PROJECT_ID}/precision-bg-repo/precision-bg-remover
```

## Troubleshooting

### Common Issues

1. **GPU Driver Issues**
```bash
# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu
```

2. **Memory Issues**
```bash
# Monitor memory usage
kubectl top pods
gcloud logging read "resource.type=cloud_run_revision" --limit=50
```

3. **Cold Start Optimization**
```bash
# Keep minimum instances warm
gcloud run services update ${SERVICE_NAME} \
    --region ${REGION} \
    --min-instances 1
```

## Performance Tuning

### 1. Model Optimization
- Use model quantization for faster inference
- Implement model caching strategies
- Optimize batch processing sizes

### 2. Infrastructure Optimization
- Use regional persistent disks for better performance
- Configure appropriate machine types for workload
- Implement CDN for static assets

### 3. Monitoring Key Metrics
- Processing latency
- Memory utilization
- GPU utilization (if applicable)
- Cost per processed image

## Conclusion

This guide provides comprehensive deployment options for Google Cloud Platform, from serverless Cloud Run to fully managed Kubernetes clusters. Choose the deployment method that best fits your performance, scalability, and cost requirements.

For production deployments, consider:
- Using Cloud Run for variable workloads
- GKE for consistent high-performance requirements
- Vertex AI for ML-focused deployments
- Proper monitoring and alerting setup
- Cost optimization strategies
- Security best practices
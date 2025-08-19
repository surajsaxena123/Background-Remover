# Microsoft Azure Deployment Guide

This comprehensive guide covers deploying the Precision Background Remover on Microsoft Azure using various services and deployment patterns.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Azure Container Instances (ACI)](#azure-container-instances-aci)
3. [Azure Kubernetes Service (AKS)](#azure-kubernetes-service-aks)
4. [Azure Container Apps](#azure-container-apps)
5. [Azure Virtual Machines](#azure-virtual-machines)
6. [Azure Machine Learning](#azure-machine-learning)
7. [Azure Functions](#azure-functions)
8. [Azure Storage Integration](#azure-storage-integration)
9. [Monitoring and Application Insights](#monitoring-and-application-insights)
10. [Cost Management](#cost-management)
11. [Security and Compliance](#security-and-compliance)

## Prerequisites

### Required Tools
- Azure CLI installed and configured
- Docker installed locally
- Azure subscription with sufficient credits/billing
- Resource group created

### Azure CLI Setup
```bash
# Install Azure CLI (if not already installed)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Set default subscription
az account set --subscription "your-subscription-id"

# Create resource group
az group create --name precision-bg-rg --location eastus

# Set environment variables
export RESOURCE_GROUP="precision-bg-rg"
export LOCATION="eastus"
export ACR_NAME="precisionbgregistry"
export SERVICE_NAME="precision-bg-remover"
```

### Enable Required Services
```bash
# Register providers
az provider register --namespace Microsoft.ContainerInstance
az provider register --namespace Microsoft.ContainerService
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.App
```

## Azure Container Instances (ACI)

Ideal for simple, serverless container deployments without orchestration overhead.

### 1. Create Azure Container Registry

```bash
# Create ACR
az acr create \
    --resource-group ${RESOURCE_GROUP} \
    --name ${ACR_NAME} \
    --sku Standard \
    --admin-enabled true

# Get login server
ACR_LOGIN_SERVER=$(az acr show --name ${ACR_NAME} --query loginServer --output tsv)
echo $ACR_LOGIN_SERVER
```

### 2. Build and Push Container Image

```bash
# Login to ACR
az acr login --name ${ACR_NAME}

# Build and push image
az acr build \
    --registry ${ACR_NAME} \
    --image precision-bg-remover:latest \
    --file Dockerfile .
```

### 3. Deploy to Azure Container Instances

```bash
# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name ${ACR_NAME} --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name ${ACR_NAME} --query passwords[0].value --output tsv)

# Create container instance
az container create \
    --resource-group ${RESOURCE_GROUP} \
    --name precision-bg-container \
    --image ${ACR_LOGIN_SERVER}/precision-bg-remover:latest \
    --cpu 2 \
    --memory 4 \
    --registry-login-server ${ACR_LOGIN_SERVER} \
    --registry-username ${ACR_USERNAME} \
    --registry-password ${ACR_PASSWORD} \
    --dns-name-label precision-bg-unique \
    --ports 8501 \
    --environment-variables \
        PRECISION_MODE=ultra_high \
        ENABLE_GPU=false \
    --restart-policy OnFailure
```

### 4. Configure Custom Domain and SSL

Create `aci-deployment.yaml`:

```yaml
apiVersion: 2021-10-01
location: eastus
name: precision-bg-container
properties:
  containers:
  - name: precision-bg-app
    properties:
      image: precisionbgregistry.azurecr.io/precision-bg-remover:latest
      resources:
        requests:
          cpu: 2.0
          memoryInGb: 4.0
      ports:
      - port: 8501
        protocol: TCP
      environmentVariables:
      - name: PRECISION_MODE
        value: ultra_high
      - name: ENABLE_GPU
        value: "false"
  imageRegistryCredentials:
  - server: precisionbgregistry.azurecr.io
    username: precisionbgregistry
    password: <registry-password>
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 8501
    dnsNameLabel: precision-bg-unique
  osType: Linux
  restartPolicy: OnFailure
tags:
  Environment: Production
  Application: PrecisionBackgroundRemover
type: Microsoft.ContainerInstance/containerGroups
```

Deploy using ARM template:
```bash
az deployment group create \
    --resource-group ${RESOURCE_GROUP} \
    --template-file aci-deployment.yaml
```

## Azure Kubernetes Service (AKS)

For production-grade containerized applications with advanced orchestration.

### 1. Create AKS Cluster with GPU Support

```bash
# Create AKS cluster
az aks create \
    --resource-group ${RESOURCE_GROUP} \
    --name precision-bg-aks \
    --node-count 2 \
    --node-vm-size Standard_D4s_v3 \
    --enable-addons monitoring \
    --generate-ssh-keys \
    --attach-acr ${ACR_NAME} \
    --enable-cluster-autoscaler \
    --min-count 1 \
    --max-count 5 \
    --location ${LOCATION}

# Add GPU node pool
az aks nodepool add \
    --resource-group ${RESOURCE_GROUP} \
    --cluster-name precision-bg-aks \
    --name gpunodepool \
    --node-count 1 \
    --node-vm-size Standard_NC6s_v3 \
    --min-count 0 \
    --max-count 3 \
    --enable-cluster-autoscaler \
    --node-taints sku=gpu:NoSchedule

# Get credentials
az aks get-credentials \
    --resource-group ${RESOURCE_GROUP} \
    --name precision-bg-aks
```

### 2. Install NVIDIA GPU Driver

```bash
# Install GPU driver DaemonSet
kubectl apply -f https://raw.githubusercontent.com/Azure/aks-engine/master/examples/addons/nvidia-device-plugin/daemonset.yaml
```

### 3. Deploy Application to AKS

Create `k8s-azure/namespace.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: precision-bg
  labels:
    name: precision-bg
```

Create `k8s-azure/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: precision-bg-deployment
  namespace: precision-bg
spec:
  replicas: 3
  selector:
    matchLabels:
      app: precision-bg-remover
  template:
    metadata:
      labels:
        app: precision-bg-remover
    spec:
      containers:
      - name: precision-bg-app
        image: precisionbgregistry.azurecr.io/precision-bg-remover:latest
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
        - name: AZURE_STORAGE_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: azure-storage-secret
              key: connection-string
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
      - key: sku
        operator: Equal
        value: gpu
        effect: NoSchedule
      nodeSelector:
        agentpool: gpunodepool
---
apiVersion: v1
kind: Service
metadata:
  name: precision-bg-service
  namespace: precision-bg
spec:
  selector:
    app: precision-bg-remover
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: precision-bg-ingress
  namespace: precision-bg
  annotations:
    kubernetes.io/ingress.class: azure/application-gateway
    appgw.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - precision-bg.yourdomain.com
    secretName: tls-secret
  rules:
  - host: precision-bg.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: precision-bg-service
            port:
              number: 80
```

Deploy to AKS:

```bash
kubectl apply -f k8s-azure/namespace.yaml
kubectl apply -f k8s-azure/deployment.yaml

# Check deployment status
kubectl get pods -n precision-bg
kubectl get services -n precision-bg
```

### 4. Set Up Horizontal Pod Autoscaler

Create `k8s-azure/hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: precision-bg-hpa
  namespace: precision-bg
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: precision-bg-deployment
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

Apply HPA:
```bash
kubectl apply -f k8s-azure/hpa.yaml
```

## Azure Container Apps

Serverless containers with built-in scaling and traffic management.

### 1. Create Container Apps Environment

```bash
# Install Container Apps extension
az extension add --name containerapp

# Create Container Apps environment
az containerapp env create \
    --name precision-bg-env \
    --resource-group ${RESOURCE_GROUP} \
    --location ${LOCATION}
```

### 2. Deploy to Container Apps

```bash
az containerapp create \
    --name precision-bg-app \
    --resource-group ${RESOURCE_GROUP} \
    --environment precision-bg-env \
    --image ${ACR_LOGIN_SERVER}/precision-bg-remover:latest \
    --registry-server ${ACR_LOGIN_SERVER} \
    --registry-username ${ACR_USERNAME} \
    --registry-password ${ACR_PASSWORD} \
    --target-port 8501 \
    --ingress external \
    --cpu 2.0 \
    --memory 4.0Gi \
    --min-replicas 0 \
    --max-replicas 10 \
    --env-vars \
        PRECISION_MODE=ultra_high \
        ENABLE_GPU=false
```

### 3. Configure Custom Domain and SSL

```bash
# Add custom domain
az containerapp hostname add \
    --hostname precision-bg.yourdomain.com \
    --name precision-bg-app \
    --resource-group ${RESOURCE_GROUP}

# Bind SSL certificate
az containerapp ssl add \
    --hostname precision-bg.yourdomain.com \
    --name precision-bg-app \
    --resource-group ${RESOURCE_GROUP} \
    --certificate-name precision-bg-cert \
    --certificate-file ./ssl-cert.pfx \
    --certificate-password "your-cert-password"
```

## Azure Virtual Machines

For custom deployments requiring full control over the environment.

### 1. Create GPU-Enabled Virtual Machine

```bash
# Create VM with GPU
az vm create \
    --resource-group ${RESOURCE_GROUP} \
    --name precision-bg-vm \
    --image UbuntuLTS \
    --size Standard_NC6s_v3 \
    --admin-username azureuser \
    --generate-ssh-keys \
    --custom-data cloud-init.yml \
    --public-ip-sku Standard \
    --nsg precision-bg-nsg

# Create NSG rules
az network nsg rule create \
    --resource-group ${RESOURCE_GROUP} \
    --nsg-name precision-bg-nsg \
    --name AllowHTTP \
    --protocol tcp \
    --priority 1000 \
    --destination-port-range 80 \
    --access allow

az network nsg rule create \
    --resource-group ${RESOURCE_GROUP} \
    --nsg-name precision-bg-nsg \
    --name AllowHTTPS \
    --protocol tcp \
    --priority 1001 \
    --destination-port-range 443 \
    --access allow

az network nsg rule create \
    --resource-group ${RESOURCE_GROUP} \
    --nsg-name precision-bg-nsg \
    --name AllowStreamlit \
    --protocol tcp \
    --priority 1002 \
    --destination-port-range 8501 \
    --access allow
```

### 2. Create Cloud-Init Configuration

Create `cloud-init.yml`:

```yaml
#cloud-config
package_upgrade: true

packages:
  - docker.io
  - nvidia-container-toolkit
  - nginx
  - certbot
  - python3-certbot-nginx

write_files:
- content: |
    server {
        listen 80;
        server_name precision-bg.yourdomain.com;
        
        location / {
            proxy_pass http://localhost:8501;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }
    }
  path: /etc/nginx/sites-available/precision-bg
  permissions: '0644'

runcmd:
  - systemctl enable docker
  - systemctl start docker
  - usermod -aG docker azureuser
  - systemctl restart docker
  - ln -s /etc/nginx/sites-available/precision-bg /etc/nginx/sites-enabled/
  - systemctl restart nginx
  - systemctl enable nginx
  - docker run -d --gpus all --name precision-bg-container -p 8501:8501 --restart unless-stopped precisionbgregistry.azurecr.io/precision-bg-remover:latest
```

### 3. Set Up SSL with Let's Encrypt

SSH into the VM and run:

```bash
# Generate SSL certificate
sudo certbot --nginx -d precision-bg.yourdomain.com --non-interactive --agree-tos -m your-email@domain.com

# Set up auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

## Azure Machine Learning

For ML-focused deployments with model management and monitoring.

### 1. Create Azure ML Workspace

```bash
# Create ML workspace
az ml workspace create \
    --name precision-bg-ml-workspace \
    --resource-group ${RESOURCE_GROUP} \
    --location ${LOCATION}
```

### 2. Create ML Compute Instance

```bash
# Create compute instance
az ml compute create \
    --name precision-bg-compute \
    --type ComputeInstance \
    --size Standard_NC6s_v3 \
    --workspace-name precision-bg-ml-workspace \
    --resource-group ${RESOURCE_GROUP}
```

### 3. Deploy Model as Web Service

Create `azure_ml_deployment.py`:

```python
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.image import ContainerImage
import os

# Connect to workspace
ws = Workspace.from_config()

# Register model
model = Model.register(
    workspace=ws,
    model_path="./models",
    model_name="precision-bg-remover",
    description="Precision background removal model"
)

# Create container image
container_config = ContainerImage.image_configuration(
    execution_script="score.py",
    runtime="python",
    conda_file="conda_env.yml",
    docker_file="Dockerfile"
)

# Deploy to ACI
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=2,
    memory_gb=4,
    description="Precision Background Remover API"
)

service = Model.deploy(
    workspace=ws,
    name="precision-bg-service",
    models=[model],
    image_config=container_config,
    deployment_config=aci_config
)

service.wait_for_deployment(show_output=True)
print(f"Service endpoint: {service.scoring_uri}")
```

Create `score.py` for Azure ML:

```python
import json
import os
import cv2
import numpy as np
import base64
from src.core import remove_background_precision_grade

def init():
    """Initialize the model."""
    global model_initialized
    model_initialized = True
    print("Model initialized successfully")

def run(raw_data):
    """Process inference request."""
    try:
        # Parse input data
        data = json.loads(raw_data)
        image_data = data['image']
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image
        result, metrics = remove_background_precision_grade(
            image,
            precision_mode=data.get('precision_mode', 'ultra_high')
        )
        
        # Encode result
        _, buffer = cv2.imencode('.png', result)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return json.dumps({
            'result_image': result_b64,
            'quality_metrics': metrics
        })
        
    except Exception as e:
        return json.dumps({'error': str(e)})
```

## Azure Functions

For event-driven, serverless background processing.

### 1. Create Function App

```bash
# Create storage account
az storage account create \
    --name precisionbgstorage \
    --resource-group ${RESOURCE_GROUP} \
    --location ${LOCATION} \
    --sku Standard_LRS

# Create Function App
az functionapp create \
    --resource-group ${RESOURCE_GROUP} \
    --consumption-plan-location ${LOCATION} \
    --runtime python \
    --runtime-version 3.9 \
    --functions-version 4 \
    --name precision-bg-functions \
    --storage-account precisionbgstorage
```

### 2. Create Blob Trigger Function

Create `function_app.py`:

```python
import azure.functions as func
import logging
import cv2
import numpy as np
from azure.storage.blob import BlobServiceClient
from src.core import remove_background_precision_grade
import os

app = func.FunctionApp()

@app.blob_trigger(arg_name="myblob", 
                 path="input-images/{name}",
                 connection="AzureWebJobsStorage")
def process_image_blob(myblob: func.InputStream):
    logging.info(f"Processing blob: {myblob.name}")
    
    try:
        # Read image data
        image_data = myblob.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image
        result, metrics = remove_background_precision_grade(image)
        
        # Save result to output blob
        _, buffer = cv2.imencode('.png', result)
        
        # Upload to output container
        blob_service_client = BlobServiceClient.from_connection_string(
            os.environ['AzureWebJobsStorage']
        )
        
        output_blob_name = f"processed_{myblob.name.split('/')[-1]}"
        blob_client = blob_service_client.get_blob_client(
            container="output-images",
            blob=output_blob_name
        )
        
        blob_client.upload_blob(buffer.tobytes(), overwrite=True)
        
        logging.info(f"Processed {myblob.name} with quality: {metrics.get('overall_quality', 0)}")
        
    except Exception as e:
        logging.error(f"Error processing {myblob.name}: {str(e)}")
```

### 3. Deploy Function App

```bash
# Deploy function
func azure functionapp publish precision-bg-functions
```

## Azure Storage Integration

### 1. Create Storage Account with Containers

```bash
# Create storage account
az storage account create \
    --name precisionbgstorage \
    --resource-group ${RESOURCE_GROUP} \
    --location ${LOCATION} \
    --sku Standard_LRS \
    --kind StorageV2

# Get connection string
STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
    --name precisionbgstorage \
    --resource-group ${RESOURCE_GROUP} \
    --output tsv)

# Create containers
az storage container create \
    --name input-images \
    --connection-string "${STORAGE_CONNECTION_STRING}"

az storage container create \
    --name output-images \
    --connection-string "${STORAGE_CONNECTION_STRING}"

az storage container create \
    --name models \
    --connection-string "${STORAGE_CONNECTION_STRING}"
```

### 2. Set Up Lifecycle Management

Create `lifecycle-policy.json`:

```json
{
  "rules": [
    {
      "name": "DeleteOldProcessedImages",
      "enabled": true,
      "type": "Lifecycle",
      "definition": {
        "filters": {
          "blobTypes": ["blockBlob"],
          "prefixMatch": ["output-images/"]
        },
        "actions": {
          "baseBlob": {
            "delete": {
              "daysAfterModificationGreaterThan": 30
            }
          }
        }
      }
    }
  ]
}
```

Apply lifecycle policy:
```bash
az storage account management-policy create \
    --account-name precisionbgstorage \
    --resource-group ${RESOURCE_GROUP} \
    --policy lifecycle-policy.json
```

## Monitoring and Application Insights

### 1. Create Application Insights

```bash
# Create Application Insights
az monitor app-insights component create \
    --app precision-bg-insights \
    --location ${LOCATION} \
    --resource-group ${RESOURCE_GROUP} \
    --application-type web

# Get instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
    --app precision-bg-insights \
    --resource-group ${RESOURCE_GROUP} \
    --query instrumentationKey \
    --output tsv)
```

### 2. Configure Application Monitoring

Create `monitoring.py`:

```python
from applicationinsights import TelemetryClient
from applicationinsights.logging import LoggingHandler
import logging
import os

# Initialize Application Insights
instrumentation_key = os.environ.get('APPINSIGHTS_INSTRUMENTATION_KEY')
tc = TelemetryClient(instrumentation_key)

# Set up logging
handler = LoggingHandler(instrumentation_key)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger(__name__)

def track_processing_metrics(processing_time, quality_score, image_size):
    """Track custom metrics."""
    tc.track_metric('ProcessingTime', processing_time)
    tc.track_metric('QualityScore', quality_score)
    tc.track_metric('ImageSize', image_size)
    
    tc.track_event('ImageProcessed', {
        'quality_score': quality_score,
        'processing_time': processing_time
    })
    
    tc.flush()
```

### 3. Set Up Alerts and Dashboards

```bash
# Create alert rule for high processing time
az monitor metrics alert create \
    --name "High Processing Time Alert" \
    --resource-group ${RESOURCE_GROUP} \
    --scopes "/subscriptions/SUBSCRIPTION_ID/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.ContainerInstance/containerGroups/precision-bg-container" \
    --condition "avg customMetrics/ProcessingTime > 60" \
    --description "Alert when processing time exceeds 60 seconds" \
    --evaluation-frequency 5m \
    --window-size 15m \
    --severity 2
```

## Cost Management

### 1. Set Up Budget Alerts

```bash
# Create budget
az consumption budget create \
    --budget-name "Precision BG Budget" \
    --amount 100 \
    --resource-group ${RESOURCE_GROUP} \
    --time-grain Monthly \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --notification \
        enabled=true \
        operator=GreaterThan \
        threshold=80 \
        contact-emails=["admin@yourdomain.com"] \
        contact-groups=[] \
        contact-roles=["Owner"]
```

### 2. Use Spot Instances for Cost Savings

```bash
# Create VMSS with spot instances
az vmss create \
    --resource-group ${RESOURCE_GROUP} \
    --name precision-bg-vmss \
    --image UbuntuLTS \
    --vm-sku Standard_NC6s_v3 \
    --priority Spot \
    --max-price 0.50 \
    --eviction-policy Deallocate \
    --instance-count 2 \
    --upgrade-policy-mode automatic
```

## Security and Compliance

### 1. Configure Azure Key Vault

```bash
# Create Key Vault
az keyvault create \
    --name precision-bg-keyvault \
    --resource-group ${RESOURCE_GROUP} \
    --location ${LOCATION} \
    --sku standard

# Add secrets
az keyvault secret set \
    --vault-name precision-bg-keyvault \
    --name "StorageConnectionString" \
    --value "${STORAGE_CONNECTION_STRING}"

az keyvault secret set \
    --vault-name precision-bg-keyvault \
    --name "AppInsightsKey" \
    --value "${INSTRUMENTATION_KEY}"
```

### 2. Set Up Azure Active Directory Authentication

```bash
# Create service principal
az ad sp create-for-rbac \
    --name precision-bg-sp \
    --role contributor \
    --scopes /subscriptions/SUBSCRIPTION_ID/resourceGroups/${RESOURCE_GROUP}

# Assign Key Vault permissions
az keyvault set-policy \
    --name precision-bg-keyvault \
    --spn SERVICE_PRINCIPAL_ID \
    --secret-permissions get list
```

### 3. Network Security

```bash
# Create virtual network
az network vnet create \
    --resource-group ${RESOURCE_GROUP} \
    --name precision-bg-vnet \
    --address-prefix 10.0.0.0/16 \
    --subnet-name default \
    --subnet-prefix 10.0.1.0/24

# Create network security group
az network nsg create \
    --resource-group ${RESOURCE_GROUP} \
    --name precision-bg-nsg

# Configure NSG rules
az network nsg rule create \
    --resource-group ${RESOURCE_GROUP} \
    --nsg-name precision-bg-nsg \
    --name AllowHTTPS \
    --protocol tcp \
    --priority 1000 \
    --destination-port-range 443 \
    --access allow
```

## Performance Optimization

### 1. Configure Auto-scaling

Create auto-scaling rules for VMSS:

```bash
az monitor autoscale create \
    --resource-group ${RESOURCE_GROUP} \
    --resource precision-bg-vmss \
    --resource-type Microsoft.Compute/virtualMachineScaleSets \
    --name precision-bg-autoscale \
    --min-count 1 \
    --max-count 5 \
    --count 2

# Scale out rule
az monitor autoscale rule create \
    --resource-group ${RESOURCE_GROUP} \
    --autoscale-name precision-bg-autoscale \
    --condition "Percentage CPU > 70 avg 5m" \
    --scale out 1

# Scale in rule  
az monitor autoscale rule create \
    --resource-group ${RESOURCE_GROUP} \
    --autoscale-name precision-bg-autoscale \
    --condition "Percentage CPU < 30 avg 5m" \
    --scale in 1
```

### 2. Content Delivery Network (CDN)

```bash
# Create CDN profile
az cdn profile create \
    --resource-group ${RESOURCE_GROUP} \
    --name precision-bg-cdn \
    --sku Standard_Microsoft

# Create CDN endpoint
az cdn endpoint create \
    --resource-group ${RESOURCE_GROUP} \
    --profile-name precision-bg-cdn \
    --name precision-bg-endpoint \
    --origin precision-bg.yourdomain.com
```

## Troubleshooting

### Common Issues and Solutions

1. **Container Registry Authentication Issues**
```bash
# Re-authenticate with ACR
az acr login --name ${ACR_NAME}

# Check ACR credentials
az acr credential show --name ${ACR_NAME}
```

2. **AKS GPU Node Issues**
```bash
# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu

# Check GPU driver pod status
kubectl get pods -n kube-system | grep nvidia
```

3. **Storage Access Issues**
```bash
# Test storage connectivity
az storage blob list \
    --container-name input-images \
    --connection-string "${STORAGE_CONNECTION_STRING}"
```

4. **Application Insights Not Receiving Data**
```bash
# Verify instrumentation key
az monitor app-insights component show \
    --app precision-bg-insights \
    --resource-group ${RESOURCE_GROUP}
```

## Best Practices

### 1. Resource Naming Conventions
- Use consistent naming patterns
- Include environment indicators (dev, staging, prod)
- Use resource group prefixes

### 2. Security
- Use managed identities instead of service principals
- Store secrets in Key Vault
- Implement network segmentation
- Enable audit logging

### 3. Cost Optimization
- Use appropriate VM sizes for workloads
- Implement auto-shutdown for development resources
- Monitor and analyze cost patterns
- Consider reserved instances for predictable workloads

### 4. Monitoring
- Set up comprehensive health checks
- Monitor key performance indicators
- Configure proper alerting thresholds
- Implement distributed tracing

## Conclusion

This guide provides comprehensive deployment options for Microsoft Azure, from simple container instances to full-featured Kubernetes clusters with GPU support. Choose the deployment method that best aligns with your performance requirements, scaling needs, and budget constraints.

For production deployments, consider:
- Azure Kubernetes Service for complex, scalable applications
- Container Apps for serverless microservices
- Virtual Machines for custom requirements
- Comprehensive monitoring and alerting
- Proper security and compliance measures
- Cost optimization strategies
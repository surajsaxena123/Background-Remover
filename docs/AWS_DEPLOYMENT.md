# AWS Deployment Guide - Precision Background Remover

## Overview

This comprehensive guide covers deploying the Precision Background Remover on AWS using various services optimized for scalability, performance, and cost-effectiveness.

## Architecture Overview

### Production Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CloudFront    │────│   ALB/API GW    │────│      ECS/EKS    │
│   (CDN/Cache)   │    │  (Load Balancer)│    │   (Containers)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │    Lambda       │              │
         │              │  (Serverless)   │              │
         │              └─────────────────┘              │
         │                       │                       │
         ▼              ┌─────────────────┐              ▼
┌─────────────────┐     │      S3         │    ┌─────────────────┐
│   S3 + CDN      │     │  (File Storage) │    │   EC2/Fargate   │
│ (Static Assets) │     └─────────────────┘    │   (Processing)  │
└─────────────────┘              │             └─────────────────┘
                                 │                       │
                        ┌─────────────────┐    ┌─────────────────┐
                        │   RDS/DynamoDB  │    │  SageMaker/GPU  │
                        │   (Database)    │    │  (ML Inference) │
                        └─────────────────┘    └─────────────────┘
```

## 🎯 Deployment Options

### Option 1: Serverless Architecture (Recommended for variable workloads)
- **AWS Lambda**: API endpoints and light processing
- **AWS Fargate**: Container-based processing
- **S3**: Image storage and results
- **API Gateway**: RESTful API interface
- **CloudFront**: Global content delivery

### Option 2: Container-based Architecture (Recommended for consistent workloads)
- **Amazon ECS/EKS**: Orchestrated containers
- **Application Load Balancer**: Traffic distribution
- **EC2**: Compute instances with GPU support
- **ElastiCache**: Session and result caching
- **RDS**: Metadata and user management

### Option 3: SageMaker Deployment (Recommended for ML-focused workflows)
- **SageMaker Endpoints**: Real-time inference
- **SageMaker Batch Transform**: Bulk processing
- **S3**: Model artifacts and data
- **Lambda**: API orchestration

## Quick Start Deployments

### 1. Serverless Deployment with Lambda

#### Prerequisites
```bash
# Install AWS CLI and configure
pip install awscli boto3
aws configure

# Install Serverless Framework
npm install -g serverless
serverless plugin install -n serverless-python-requirements
```

#### Create `serverless.yml`
```yaml
# serverless.yml
service: precision-background-remover

provider:
  name: aws
  runtime: python3.9
  region: us-east-1
  timeout: 900  # 15 minutes for processing
  memorySize: 3008  # Maximum Lambda memory
  environment:
    S3_BUCKET: ${self:custom.s3Bucket}
    PRECISION_LEVEL: ultra_high
  
  iamRoleStatements:
    - Effect: Allow
      Action:
        - s3:GetObject
        - s3:PutObject
        - s3:DeleteObject
      Resource:
        - "arn:aws:s3:::${self:custom.s3Bucket}/*"

custom:
  s3Bucket: precision-bg-remover-${self:provider.stage}
  pythonRequirements:
    dockerizePip: true
    dockerImage: public.ecr.aws/sam/build-python3.9:latest

functions:
  processImage:
    handler: lambda_handler.process_image
    events:
      - http:
          path: /process
          method: post
          cors: true
    layers:
      - arn:aws:lambda:us-east-1:770693421928:layer:Klayers-p39-opencv-python:1
    
  healthCheck:
    handler: lambda_handler.health_check
    events:
      - http:
          path: /health
          method: get
          cors: true

resources:
  Resources:
    S3Bucket:
      Type: AWS::S3::Bucket
      Properties:
        BucketName: ${self:custom.s3Bucket}
        CorsConfiguration:
          CorsRules:
            - AllowedOrigins: ['*']
              AllowedMethods: [GET, POST, PUT]
              AllowedHeaders: ['*']

plugins:
  - serverless-python-requirements
```

#### Lambda Handler (`lambda_handler.py`)
```python
import json
import boto3
import base64
import numpy as np
import cv2
from io import BytesIO
import logging

# Import our precision background remover
from demo_working import remove_background_enhanced

s3_client = boto3.client('s3')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def process_image(event, context):
    """Process image with precision background removal"""
    
    try:
        # Parse request
        body = json.loads(event['body'])
        image_data = base64.b64decode(body['image'])
        precision_level = body.get('precision_level', 'ultra_high')
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid image format'})
            }
        
        # Process with precision background removal
        result, metrics = remove_background_enhanced(image, model='birefnet-general')
        
        if result is None:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Processing failed'})
            }
        
        # Encode result
        _, buffer = cv2.imencode('.png', result)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Store in S3 if large
        if len(result_base64) > 5 * 1024 * 1024:  # 5MB limit
            bucket = os.environ['S3_BUCKET']
            key = f"results/{context.aws_request_id}.png"
            
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=buffer.tobytes(),
                ContentType='image/png'
            )
            
            result_url = f"https://{bucket}.s3.amazonaws.com/{key}"
            
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'success': True,
                    'result_url': result_url,
                    'metrics': metrics
                })
            }
        else:
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'success': True,
                    'image': result_base64,
                    'metrics': metrics
                })
            }
            
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def health_check(event, context):
    """Health check endpoint"""
    return {
        'statusCode': 200,
        'body': json.dumps({
            'status': 'healthy',
            'timestamp': int(time.time())
        })
    }
```

#### Deploy
```bash
# Package requirements
echo "opencv-python-headless
numpy
Pillow
rembg
pymatting" > requirements.txt

# Deploy
serverless deploy --stage prod

# Test
curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/process \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'
```

### 2. Container Deployment with ECS

#### Dockerfile for AWS
```dockerfile
# Dockerfile.aws
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies
RUN yum update -y && \
    yum install -y gcc g++ cmake pkgconfig libGL-dev && \
    yum clean all

# Copy requirements and install
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD ["lambda_handler.process_image"]
```

#### ECS Task Definition
```json
{
  "family": "precision-bg-remover",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "precision-bg-remover",
      "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/precision-bg-remover:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "STREAMLIT_SERVER_HEADLESS",
          "value": "true"
        },
        {
          "name": "STREAMLIT_SERVER_PORT",
          "value": "8501"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/precision-bg-remover",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Deploy ECS Service
```bash
# Build and push image
aws ecr create-repository --repository-name precision-bg-remover
docker build -f Dockerfile.aws -t precision-bg-remover .
docker tag precision-bg-remover:latest ACCOUNT.dkr.ecr.REGION.amazonaws.com/precision-bg-remover:latest
aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/precision-bg-remover:latest

# Create ECS cluster
aws ecs create-cluster --cluster-name precision-bg-remover-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster precision-bg-remover-cluster \
  --service-name precision-bg-remover-service \
  --task-definition precision-bg-remover:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-abcdef],assignPublicIp=ENABLED}"
```

### 3. SageMaker Deployment

#### Model Package (`model.py`)
```python
import torch
import numpy as np
import cv2
from io import BytesIO
import tarfile
import os

class PrecisionBackgroundRemover:
    def __init__(self, model_dir="/opt/ml/model"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()
    
    def _load_models(self):
        """Load precision background removal models"""
        # Load your models here
        pass
    
    def predict(self, input_data):
        """Prediction function for SageMaker"""
        try:
            # Decode image
            image_data = input_data['image']
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process with precision background removal
            from demo_working import remove_background_enhanced
            result, metrics = remove_background_enhanced(image)
            
            # Encode result
            _, buffer = cv2.imencode('.png', result)
            
            return {
                'image': buffer.tobytes(),
                'metrics': metrics
            }
            
        except Exception as e:
            return {'error': str(e)}

def model_fn(model_dir):
    """Load model for SageMaker"""
    return PrecisionBackgroundRemover(model_dir)

def input_fn(request_body, request_content_type):
    """Parse input for SageMaker"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction with SageMaker"""
    return model.predict(input_data)

def output_fn(prediction, content_type):
    """Format output for SageMaker"""
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
```

#### Deploy to SageMaker
```python
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

# Package model
def create_model_package():
    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("model.py")
        tar.add("requirements.txt")
        tar.add("demo_working.py")
        tar.add("src/")

# Upload to S3
s3 = boto3.client('s3')
bucket = 'your-sagemaker-bucket'
s3.upload_file('model.tar.gz', bucket, 'precision-bg-remover/model.tar.gz')

# Create SageMaker model
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

model = PyTorchModel(
    entry_point='model.py',
    model_data=f's3://{bucket}/precision-bg-remover/model.tar.gz',
    role=role,
    framework_version='1.12.0',
    py_version='py39',
    instance_type='ml.g4dn.xlarge'  # GPU instance
)

# Deploy endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge',
    endpoint_name='precision-bg-remover-endpoint'
)
```

## Infrastructure as Code (Terraform)

### Complete AWS Infrastructure
```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  default     = "prod"
}

# VPC and Networking
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "precision-bg-remover-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "public" {
  count = 2
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true

  tags = {
    Name = "precision-bg-remover-public-${count.index + 1}"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "precision-bg-remover-igw"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "precision-bg-remover-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)
  
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# S3 Bucket for images
resource "aws_s3_bucket" "images" {
  bucket = "precision-bg-remover-images-${random_id.bucket_suffix.hex}"
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_cors_configuration" "images" {
  bucket = aws_s3_bucket.images.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "POST", "PUT"]
    allowed_origins = ["*"]
    max_age_seconds = 3000
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "precision-bg-remover-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "app" {
  family                   = "precision-bg-remover"
  requires_compatibilities = ["FARGATE"]
  network_mode            = "awsvpc"
  cpu                     = "2048"
  memory                  = "8192"
  execution_role_arn      = aws_iam_role.ecs_execution_role.arn
  task_role_arn          = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name  = "precision-bg-remover"
      image = "${aws_ecr_repository.app.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8501
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "STREAMLIT_SERVER_HEADLESS"
          value = "true"
        },
        {
          name  = "S3_BUCKET"
          value = aws_s3_bucket.images.id
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.app.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

# ECR Repository
resource "aws_ecr_repository" "app" {
  name                 = "precision-bg-remover"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ECS Service
resource "aws_ecs_service" "app" {
  name            = "precision-bg-remover-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  launch_type     = "FARGATE"
  desired_count   = 2

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "precision-bg-remover"
    container_port   = 8501
  }

  depends_on = [aws_lb_listener.app]
}

# Application Load Balancer
resource "aws_lb" "app" {
  name               = "precision-bg-remover-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = aws_subnet.public[*].id

  enable_deletion_protection = false
}

resource "aws_lb_target_group" "app" {
  name     = "precision-bg-remover-tg"
  port     = 8501
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 60
    interval            = 120
    path               = "/_stcore/health"
    matcher            = "200"
  }
}

resource "aws_lb_listener" "app" {
  load_balancer_arn = aws_lb.app.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# Security Groups
resource "aws_security_group" "alb" {
  name        = "precision-bg-remover-alb-sg"
  description = "ALB Security Group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "app" {
  name        = "precision-bg-remover-app-sg"
  description = "App Security Group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 8501
    to_port         = 8501
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# IAM Roles
resource "aws_iam_role" "ecs_execution_role" {
  name = "precision-bg-remover-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_role_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task_role" {
  name = "precision-bg-remover-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "precision-bg-remover-ecs-task-policy"
  role = aws_iam_role.ecs_task_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.images.arn}/*"
      }
    ]
  })
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/precision-bg-remover"
  retention_in_days = 7
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# Outputs
output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.app.dns_name
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.images.id
}

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.app.repository_url
}
```

## Monitoring and Observability

### CloudWatch Dashboards
```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ECS", "CPUUtilization", "ServiceName", "precision-bg-remover-service"],
          [".", "MemoryUtilization", ".", "."]
        ],
        "period": 300,
        "stat": "Average",
        "region": "us-east-1",
        "title": "ECS Resource Utilization"
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", "app/precision-bg-remover-alb"],
          [".", "TargetResponseTime", ".", "."]
        ],
        "period": 300,
        "stat": "Sum",
        "region": "us-east-1",
        "title": "Load Balancer Metrics"
      }
    }
  ]
}
```

### Custom Metrics
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

def publish_processing_metrics(processing_time, quality_score, image_size):
    """Publish custom metrics to CloudWatch"""
    
    cloudwatch.put_metric_data(
        Namespace='PrecisionBackgroundRemover',
        MetricData=[
            {
                'MetricName': 'ProcessingTime',
                'Value': processing_time,
                'Unit': 'Seconds',
                'Dimensions': [
                    {
                        'Name': 'ImageSize',
                        'Value': f"{image_size[0]}x{image_size[1]}"
                    }
                ]
            },
            {
                'MetricName': 'QualityScore',
                'Value': quality_score,
                'Unit': 'None'
            }
        ]
    )
```

## Cost Optimization

### Estimated Monthly Costs

| Component | Configuration | Monthly Cost (USD) |
|-----------|--------------|-------------------|
| **Serverless (Lambda)** | 1M requests, 30s avg execution | $200-400 |
| **Container (Fargate)** | 2 tasks, 2 vCPU, 8GB RAM | $120-180 |
| **GPU Instance (SageMaker)** | ml.g4dn.xlarge, 8h/day | $300-500 |
| **Storage (S3)** | 1TB images, 100GB results | $25-35 |
| **Load Balancer** | Application LB with 2 AZs | $20-25 |
| **Data Transfer** | 500GB outbound | $45-50 |

### Cost Optimization Strategies
1. **Use Spot Instances** for batch processing (up to 90% savings)
2. **Implement caching** with CloudFront and ElastiCache
3. **Auto-scaling** based on demand patterns
4. **Reserved Instances** for predictable workloads
5. **Lifecycle policies** for S3 storage classes

## Security Best Practices

### Security Configuration
```yaml
# security-config.yml
VPC:
  EnableDnsHostnames: true
  EnableDnsSupport: true
  FlowLogsRole: !Ref VPCFlowLogsRole

SecurityGroups:
  WebTierSG:
    - Protocol: HTTPS
      Port: 443
      Source: 0.0.0.0/0
    - Protocol: HTTP
      Port: 80
      Source: 0.0.0.0/0
      
  AppTierSG:
    - Protocol: TCP
      Port: 8501
      Source: !Ref WebTierSG
      
  DatabaseSG:
    - Protocol: TCP
      Port: 5432
      Source: !Ref AppTierSG

IAM:
  PrincipleOfLeastPrivilege: true
  MFA: required
  PasswordPolicy:
    MinLength: 14
    RequireSymbols: true
    RequireNumbers: true
    RequireUppercase: true
    RequireLowercase: true

Encryption:
  S3: AES256
  EBS: Enabled
  RDS: Enabled
  Transit: TLS1.2+
```

## CI/CD Pipeline

### GitHub Actions Deployment
```yaml
# .github/workflows/deploy.yml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build and push Docker image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: precision-bg-remover
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
    
    - name: Update ECS service
      run: |
        aws ecs update-service \
          --cluster precision-bg-remover-cluster \
          --service precision-bg-remover-service \
          --force-new-deployment
```

## Global Deployment

### Multi-Region Setup
```bash
# Deploy to multiple regions
regions=("us-east-1" "eu-west-1" "ap-southeast-1")

for region in "${regions[@]}"; do
  echo "Deploying to $region..."
  
  # Deploy infrastructure
  terraform workspace select $region || terraform workspace new $region
  terraform apply -var="aws_region=$region" -auto-approve
  
  # Deploy application
  aws ecs update-service \
    --region $region \
    --cluster precision-bg-remover-cluster \
    --service precision-bg-remover-service \
    --force-new-deployment
done
```

## Deployment Checklist

### Pre-Deployment
- [ ] AWS CLI configured with appropriate permissions
- [ ] Docker installed and configured
- [ ] Terraform installed (for IaC deployment)
- [ ] ECR repository created
- [ ] S3 bucket for artifacts created
- [ ] IAM roles and policies configured

### Deployment Steps
- [ ] Build and push Docker image to ECR
- [ ] Deploy infrastructure using Terraform
- [ ] Configure monitoring and logging
- [ ] Set up auto-scaling policies
- [ ] Configure SSL certificates
- [ ] Test deployment with sample images
- [ ] Configure backup and disaster recovery

### Post-Deployment
- [ ] Load testing with realistic traffic
- [ ] Monitor performance metrics
- [ ] Set up alerting and notifications
- [ ] Document operational procedures
- [ ] Train operations team

## Troubleshooting

### Common Issues

**Lambda Timeout**
```bash
# Increase timeout for large images
aws lambda update-function-configuration \
  --function-name precision-bg-remover \
  --timeout 900
```

**Memory Issues**
```bash
# Increase memory allocation
aws lambda update-function-configuration \
  --function-name precision-bg-remover \
  --memory-size 3008
```

**ECR Push Failures**
```bash
# Re-authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-east-1.amazonaws.com
```

**ECS Service Not Starting**
```bash
# Check service events
aws ecs describe-services \
  --cluster precision-bg-remover-cluster \
  --services precision-bg-remover-service
```

### Performance Optimization

**Optimize Cold Starts (Lambda)**
```python
# Pre-load models outside handler
import torch
model = torch.jit.load('model.pt')

def lambda_handler(event, context):
    # Use pre-loaded model
    return process_with_model(model, image)
```

**Container Optimization**
```dockerfile
# Multi-stage build for smaller images
FROM python:3.9-slim as builder
RUN pip install --user rembg torch

FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
```

## Cost Optimization Guide

### Estimated Monthly Costs by Usage

| Usage Tier | Images/Month | Lambda Cost | S3 Cost | Data Transfer | Total |
|------------|--------------|-------------|---------|---------------|-------|
| **Starter** | 1,000 | $15 | $5 | $5 | $25 |
| **Professional** | 10,000 | $150 | $25 | $25 | $200 |
| **Enterprise** | 100,000 | $800 | $100 | $150 | $1,050 |
| **Scale** | 1,000,000 | $6,000 | $500 | $1,000 | $7,500 |

### Cost Optimization Strategies

**1. Use Spot Instances for Batch Processing**
```yaml
# ECS with Spot instances
CapacityProviders:
  - Name: spot-capacity-provider
    AutoScalingGroupProvider:
      AutoScalingGroupArn: !Ref SpotAutoScalingGroup
      ManagedScaling:
        Status: ENABLED
        TargetCapacity: 80
```

**2. Implement Intelligent Caching**
```python
import hashlib
import boto3

def get_cache_key(image_data):
    return hashlib.md5(image_data).hexdigest()

def check_cache(cache_key):
    s3 = boto3.client('s3')
    try:
        return s3.get_object(Bucket='cache-bucket', Key=cache_key)
    except:
        return None
```

**3. Auto-scaling Based on Queue Depth**
```json
{
  "MetricName": "ApproximateNumberOfMessages",
  "Namespace": "AWS/SQS",
  "Statistic": "Average",
  "Dimensions": [
    {
      "Name": "QueueName",
      "Value": "precision-bg-processing-queue"
    }
  ]
}
```

## Security Implementation

### Network Security
```hcl
# WAF Configuration
resource "aws_wafv2_web_acl" "precision_bg_waf" {
  name  = "precision-bg-remover-waf"
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  rule {
    name     = "rate-limit"
    priority = 1

    override_action {
      none {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "rateLimitRule"
      sampled_requests_enabled   = true
    }
  }
}
```

### Data Encryption
```python
import boto3
from cryptography.fernet import Fernet

class SecureImageProcessor:
    def __init__(self):
        self.kms = boto3.client('kms')
        self.key_id = 'arn:aws:kms:us-east-1:123456789012:key/key-id'
    
    def encrypt_image(self, image_data):
        response = self.kms.encrypt(
            KeyId=self.key_id,
            Plaintext=image_data
        )
        return response['CiphertextBlob']
    
    def decrypt_image(self, encrypted_data):
        response = self.kms.decrypt(CiphertextBlob=encrypted_data)
        return response['Plaintext']
```

### API Security
```yaml
# API Gateway with authentication
Resources:
  PrecisionBgApi:
    Type: AWS::ApiGateway::RestApi
    Properties:
      Name: precision-bg-remover-api
      EndpointConfiguration:
        Types:
          - REGIONAL
      Policy:
        Statement:
          - Effect: Allow
            Principal: "*"
            Action: execute-api:Invoke
            Resource: "*"
            Condition:
              IpAddress:
                aws:SourceIp:
                  - "203.0.113.0/24"  # Allowed IP ranges
```

## Multi-Region Deployment

### Global Distribution Strategy
```bash
#!/bin/bash
# Deploy to multiple regions for global availability

REGIONS=("us-east-1" "eu-west-1" "ap-southeast-1" "us-west-2")

for region in "${REGIONS[@]}"; do
  echo "Deploying to $region..."
  
  # Set region-specific variables
  export AWS_DEFAULT_REGION=$region
  export TF_VAR_region=$region
  
  # Deploy infrastructure
  terraform workspace select $region || terraform workspace new $region
  terraform apply -auto-approve
  
  # Deploy application
  aws ecs update-service \
    --cluster precision-bg-remover-cluster \
    --service precision-bg-remover-service \
    --force-new-deployment \
    --region $region
done
```

### Route 53 Global Load Balancing
```hcl
resource "aws_route53_health_check" "precision_bg_health" {
  for_each = var.regions
  
  fqdn                            = aws_lb.app[each.key].dns_name
  port                            = 80
  type                            = "HTTP"
  resource_path                   = "/health"
  failure_threshold               = "3"
  request_interval                = "30"
  
  tags = {
    Name = "precision-bg-${each.key}-health-check"
  }
}

resource "aws_route53_record" "precision_bg_global" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "api.precision-bg-remover.com"
  type    = "A"
  
  set_identifier = "global"
  
  geolocation_routing_policy {
    continent = "NA"
  }
  
  alias {
    name                   = aws_lb.app["us-east-1"].dns_name
    zone_id               = aws_lb.app["us-east-1"].zone_id
    evaluate_target_health = true
  }
}
```

This comprehensive AWS deployment guide provides multiple deployment strategies for the Precision Background Remover, from serverless to container-based solutions, with complete infrastructure automation, monitoring, security best practices, cost optimization strategies, and global deployment capabilities.
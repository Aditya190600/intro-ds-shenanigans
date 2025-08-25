#!/bin/bash

# Build and Deploy Weather Analysis App to K3s

set -e

echo "Building Docker image..."
docker build -t weather-analysis:latest .

# Import image to K3s (if using K3s with docker)
echo "Importing image to K3s..."
docker save weather-analysis:latest | sudo k3s ctr images import -

# Apply Kubernetes manifests
echo "Deploying to K3s..."
kubectl apply -k k8s/

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/weather-analysis-app

# Get service info
echo "Deployment complete!"
echo "Service information:"
kubectl get services weather-analysis-service
echo ""
echo "Pod information:"
kubectl get pods -l app=weather-analysis
echo ""
echo "To access the app, you can port-forward:"
echo "kubectl port-forward service/weather-analysis-service 8501:80"
#!/bin/bash

# Setup script for Confidential GKE cluster with Altastata Jupyter
# This script creates a GKE cluster with confidential nodes and deploys the Jupyter container

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"altastata-coco"}
CLUSTER_NAME=${CLUSTER_NAME:-"altastata-confidential-cluster"}
ZONE=${ZONE:-"us-central1-a"}
MACHINE_TYPE=${MACHINE_TYPE:-"n2d-standard-4"}
NODE_COUNT=${NODE_COUNT:-1}
BUCKET_NAME=${BUCKET_NAME:-"altastata-confidential-storage"}
SERVICE_ACCOUNT_NAME=${SERVICE_ACCOUNT_NAME:-"jupyter-storage"}

echo "🚀 Setting up Confidential GKE cluster for Altastata Jupyter..."

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Install gke-gcloud-auth-plugin if not already installed
echo "🔧 Installing gke-gcloud-auth-plugin..."
gcloud components install gke-gcloud-auth-plugin --quiet || echo "Plugin may already be installed"

# Set the project
echo "📋 Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable container.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable iam.googleapis.com

# Note: Altastata handles its own storage configuration
echo "ℹ️  Altastata will handle storage configuration internally"

# Create GKE cluster with confidential nodes (if it doesn't exist)
echo "🏗️ Creating Confidential GKE cluster..."
if gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE --quiet &>/dev/null; then
    echo "ℹ️  Cluster $CLUSTER_NAME already exists, skipping creation"
else
    gcloud container clusters create $CLUSTER_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --enable-confidential-nodes \
        --num-nodes=$NODE_COUNT \
        --image-type=COS_CONTAINERD \
        --enable-autoscaling \
        --min-nodes=1 \
        --max-nodes=3 \
        --enable-autorepair \
        --enable-autoupgrade \
        --workload-pool=$PROJECT_ID.svc.id.goog
fi

# Get cluster credentials
echo "🔐 Getting cluster credentials..."
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE

# Note: Cluster already has confidential nodes enabled
echo "ℹ️  Cluster already has confidential nodes enabled"

# Note: No secrets needed - Altastata handles storage internally
echo "ℹ️  No Kubernetes secrets needed - Altastata handles storage internally"

# Deploy the Jupyter container
echo "🚀 Deploying Jupyter container..."
kubectl apply -f jupyter-deployment.yaml

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/altastata-jupyter-confidential

# Get the external IP
echo "🌐 Getting external IP..."
EXTERNAL_IP=$(kubectl get service altastata-jupyter-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -z "$EXTERNAL_IP" ]; then
    echo "⏳ External IP is still being assigned. Please wait a moment and run:"
    echo "kubectl get service altastata-jupyter-service"
else
    echo "✅ Jupyter Lab is available at: http://$EXTERNAL_IP:8888"
fi

# Display useful commands
echo ""
echo "📋 Useful commands:"
echo "  View logs: kubectl logs -f deployment/altastata-jupyter-confidential"
echo "  Get service IP: kubectl get service altastata-jupyter-service"
echo "  Access container: kubectl exec -it deployment/altastata-jupyter-confidential -- /bin/bash"
echo "  Delete cluster: gcloud container clusters delete $CLUSTER_NAME --zone=$ZONE"
echo ""
echo "🔧 Multi-architecture image deployed successfully"
echo "📝 Access Jupyter Lab to configure Altastata for your storage needs"

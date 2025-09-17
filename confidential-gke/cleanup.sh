#!/bin/bash

# Cleanup script for Confidential GKE cluster
# This script removes the cluster and associated resources

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"altastata-coco"}
CLUSTER_NAME=${CLUSTER_NAME:-"altastata-confidential-cluster"}
ZONE=${ZONE:-"us-central1-a"}
SERVICE_ACCOUNT_NAME=${SERVICE_ACCOUNT_NAME:-"jupyter-storage"}

echo "🧹 Cleaning up Confidential GKE cluster and resources..."

# Set the project
gcloud config set project $PROJECT_ID

# Delete the GKE cluster
echo "🗑️ Deleting GKE cluster..."
gcloud container clusters delete $CLUSTER_NAME --zone=$ZONE --quiet || echo "Cluster may not exist"

# Delete the service account
echo "👤 Deleting service account..."
gcloud iam service-accounts delete $SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com --quiet || echo "Service account may not exist"

# Note: No service account key file to remove (Altastata handles storage internally)

# Clean up local files
echo "📁 Cleaning up local files..."
rm -f jupyter-deployment.yaml.bak

echo "✅ Cleanup completed!"
echo ""
echo "💡 Note: GCS buckets and other persistent resources were not deleted"
echo "   You may want to manually clean up:"
echo "   - GCS buckets: gsutil rm -r gs://your-bucket-name"
echo "   - BigQuery datasets: bq rm -r dataset_name"

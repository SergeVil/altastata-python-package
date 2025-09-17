# Altastata Jupyter in Confidential GKE

Simple deployment for running your Altastata Jupyter DataScience container in Google Cloud Platform's Confidential GKE environment with cloud storage connectivity.

## Overview

This setup provides:
- **Confidential Computing**: Your data remains encrypted in memory during processing
- **Cloud Storage Connectivity**: Support for GCP, AWS S3, and Azure Blob Storage
- **Security**: Hardware-based security with AMD SEV/Intel TDX
- **Altastata Integration**: Ready for Altastata JAR-based storage configuration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Confidential GKE Node                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Jupyter Container                      │   │
│  │  ┌─────────────────┐  ┌─────────────────────────┐  │   │
│  │  │   Jupyter Lab   │  │   Altastata Package     │  │   │
│  │  │   (Port 8888)   │  │   - Storage Management  │  │   │
│  │  │                 │  │   - Data Processing     │  │   │
│  │  └─────────────────┘  │   - Security            │  │   │
│  │                       └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Cloud Storage   │
                    │ - GCP Storage   │
                    │ - AWS S3        │
                    │ - Azure Blob    │
                    └─────────────────┘
```

## Quick Reference

| Action | Command |
|--------|---------|
| **Access Jupyter** | `http://34.66.100.250:8888/lab` |
| **Stop Container** | `kubectl scale deployment altastata-jupyter-confidential --replicas=0` |
| **Start Container** | `kubectl scale deployment altastata-jupyter-confidential --replicas=1` |
| **Check Status** | `kubectl get pods -l app=altastata-jupyter` |
| **View Logs** | `kubectl logs -f deployment/altastata-jupyter-confidential` |
| **Stop Cluster** | `gcloud container clusters delete altastata-confidential-cluster --zone=us-central1-a` |
| **Start Cluster** | `./setup-cluster.sh` (recreate cluster) |

## Files

- `jupyter-deployment.yaml` - Simple Kubernetes deployment configuration
- `setup-cluster.sh` - Automated cluster setup script
- `cleanup.sh` - Cleanup script for removing resources
- `notebook-examples/altastata-azure.ipynb` - Example notebook for Azure Blob Storage connectivity

## Prerequisites

### Required Services
- Google Cloud SDK installed and authenticated
- kubectl installed
- gke-gcloud-auth-plugin installed
- Docker (for local testing)
- GCP Project with billing enabled

### Required APIs
- Google Kubernetes Engine API
- Google Cloud Storage API
- Compute Engine API
- IAM API

### Required Credentials
- Service account with appropriate permissions
- Cloud storage access credentials

## Quick Start

### 1. Configure Environment

```bash
# Set your GCP project ID
export GCP_PROJECT_ID="altastata-coco"

# Optional: Customize other settings
export CLUSTER_NAME="altastata-confidential-cluster"
export ZONE="us-central1-a"
export BUCKET_NAME="altastata-confidential-storage"
export SERVICE_ACCOUNT_NAME="jupyter-storage"
```

### 2. Install Required Tools

```bash
# Install gke-gcloud-auth-plugin (required for kubectl with GKE)
gcloud components install gke-gcloud-auth-plugin

# Verify kubectl is installed
kubectl version --client
```

### 3. Enable Required APIs

```bash
# Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable iam.googleapis.com
```

### 4. Deploy the Cluster

```bash
# Make scripts executable
chmod +x setup-cluster.sh cleanup.sh

# Run the setup script
./setup-cluster.sh
```

This will:
- Create a Confidential GKE cluster with AMD SEV security
- Deploy your multi-architecture Jupyter container
- Configure basic storage connectivity
- Set up load balancer for external access

### 5. Access Jupyter Lab

After deployment, get the external IP:

```bash
kubectl get service altastata-jupyter-service
```

**Current deployment URL**: `http://34.66.100.250:8888/lab`

Open your browser to: `http://EXTERNAL_IP:8888/lab`

> **Note**: The `/lab` path is important for accessing Jupyter Lab interface

### 6. Test Connectivity

Open the example notebook `notebook-examples/altastata-azure.ipynb` to:
- Check cloud storage credentials
- Test Altastata package import
- Verify confidential computing features
- Configure Altastata for your use case

## Cluster Management

### Check Cluster Status

```bash
# Check cluster status
kubectl get pods,services

# Check cluster details
kubectl describe deployment altastata-jupyter-confidential
```

### Access Container

```bash
# Interactive shell
kubectl exec -it deployment/altastata-jupyter-confidential -- /bin/bash

# Check container status
kubectl exec -it deployment/altastata-jupyter-confidential -- ps aux
```

### Container Management

#### Stop/Start Container (Pod Level)
```bash
# Stop the container (scale to 0 replicas)
kubectl scale deployment altastata-jupyter-confidential --replicas=0

# Start the container (scale to 1 replica)
kubectl scale deployment altastata-jupyter-confidential --replicas=1

# Check container status
kubectl get pods -l app=altastata-jupyter
```

#### Scale Resources
```bash
# Scale up/down
kubectl scale deployment altastata-jupyter-confidential --replicas=2

# Update resource limits
kubectl edit deployment altastata-jupyter-confidential
```

#### Restart Container
```bash
# Restart the deployment
kubectl rollout restart deployment/altastata-jupyter-confidential

# Check rollout status
kubectl rollout status deployment/altastata-jupyter-confidential
```

## Altastata Storage Management

### Environment Variables

The container can be configured with environment variables that Altastata uses for cloud storage access:

- **GCP**: `GOOGLE_APPLICATION_CREDENTIALS` (service account key file)
- **AWS**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
- **Azure**: `AZURE_STORAGE_CONNECTION_STRING`

**Note**: Altastata handles storage configuration internally, so these environment variables are optional and only needed if you want to use specific cloud storage providers.

### Setting Up Credentials

#### GCP Credentials
```bash
# Note: GCP credentials are optional since Altastata handles storage internally
# If you need GCP access, you can set environment variables:
kubectl set env deployment/altastata-jupyter-confidential \
    GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

#### AWS Credentials
```bash
# Set AWS credentials in deployment
kubectl set env deployment/altastata-jupyter-confidential \
    AWS_ACCESS_KEY_ID=your-access-key \
    AWS_SECRET_ACCESS_KEY=your-secret-key \
    AWS_DEFAULT_REGION=us-east-1
```

#### Azure Credentials
```bash
# Set Azure connection string
kubectl set env deployment/altastata-jupyter-confidential \
    AZURE_STORAGE_CONNECTION_STRING="your-connection-string"
```


### Altastata Integration

Altastata handles all storage management and uses these credentials automatically:

```python
from altastata import AltaStataFunctions

# Altastata automatically uses environment variables for storage access
# Configure your specific storage settings in Altastata
altastata_functions = AltaStataFunctions.from_credentials(
    user_properties, private_key
)

# Altastata manages all storage operations
# No additional storage manager needed
```

## Security Features

### Confidential Computing
- **Memory Encryption**: Data remains encrypted in memory during processing
- **Hardware Security**: Uses AMD SEV or Intel TDX technology
- **Zero Trust**: Even cloud administrators cannot access your data

### Access Control
- **Service Account**: Dedicated service account with minimal permissions
- **IAM Integration**: Proper role-based access control
- **Secret Management**: Secure credential handling

### Data Protection
- **Encryption in Transit**: All data transfers are encrypted
- **Encryption at Rest**: Data stored in GCS is encrypted
- **Secure Processing**: Data processing happens in encrypted memory

## Cost Management

### Resource Sizing
- Start with smaller instances and scale up as needed
- Use preemptible nodes for development
- Set up auto-scaling based on usage

### Cost Optimization
- Stop the cluster when not in use
- Use appropriate storage classes
- Monitor resource usage

### Stop/Start Cluster
```bash
# Delete cluster (save money - GKE clusters can't be stopped, only deleted)
gcloud container clusters delete altastata-confidential-cluster --zone=us-central1-a

# Recreate cluster (when needed)
./setup-cluster.sh
```

**Note**: GKE clusters cannot be "stopped" - they must be deleted to avoid charges. Use `./setup-cluster.sh` to recreate when needed.

### Cost Comparison
| Usage Pattern | Monthly Cost | Savings |
|---------------|--------------|---------|
| **24/7 Running** | ~$200 | - |
| **Weekdays Only** | ~$50 | 75% |
| **4 hours/day** | ~$25 | 88% |
| **Weekends Off** | ~$150 | 25% |

## Troubleshooting

### Common Issues

1. **Container not starting**
   ```bash
   kubectl describe pod -l app=altastata-jupyter
   kubectl logs -l app=altastata-jupyter
   ```

2. **Storage access issues**
   ```bash
   # Check environment variables
   kubectl exec -it deployment/altastata-jupyter-confidential -- env | grep -E "(GOOGLE|AWS|AZURE)"
   
   # Verify Altastata package
   kubectl exec -it deployment/altastata-jupyter-confidential -- python -c "import altastata; print('Altastata package loaded successfully')"
   ```

3. **Storage connectivity issues**
   ```bash
   # Check environment variables
   kubectl exec -it deployment/altastata-jupyter-confidential -- env | grep -E "(GOOGLE|AWS|AZURE)"
   
   # Check Altastata package
   kubectl exec -it deployment/altastata-jupyter-confidential -- python -c "import altastata; print('OK')"
   ```

4. **Cluster creation issues**
   ```bash
   # Check API enablement
   gcloud services list --enabled --filter="name:container.googleapis.com"
   
   # Check service account permissions
   gcloud iam service-accounts get-iam-policy jupyter-storage@your-project-id.iam.gserviceaccount.com
   ```

5. **kubectl authentication issues**
   ```bash
   # Install gke-gcloud-auth-plugin if missing
   gcloud components install gke-gcloud-auth-plugin
   
   # Verify plugin is installed
   gcloud components list --filter="name:gke-gcloud-auth-plugin"
   
   # Re-authenticate if needed
   gcloud auth login
   gcloud auth application-default login
   ```

6. **Confidential computing issues**
   ```bash
   # Check if confidential nodes are enabled
   gcloud container clusters describe altastata-confidential-cluster --zone=us-central1-a --format="value(nodeConfig.confidentialNodes.enabled)"
   
   # Check node pool configuration
   gcloud container node-pools list --cluster=altastata-confidential-cluster --zone=us-central1-a
   ```

### Verification Commands

```bash
# Check cluster status
kubectl get pods,services

# Check cluster details
kubectl describe deployment altastata-jupyter-confidential

# Check logs
kubectl logs -f deployment/altastata-jupyter-confidential

# Check environment variables
kubectl exec -it deployment/altastata-jupyter-confidential -- env | grep -E "(GOOGLE|AWS|AZURE)"

# Check Altastata package
kubectl exec -it deployment/altastata-jupyter-confidential -- python -c "import altastata; print('OK')"
```

### Performance Optimization

1. **Resource Limits**: Adjust CPU/memory limits in deployment
2. **Storage Class**: Use SSD persistent disks for better performance
3. **Caching**: Use local storage for frequently accessed data
4. **Network**: Ensure good network connectivity to cloud storage

## Cleanup

To remove all resources:

```bash
./cleanup.sh
```

This will delete:
- GKE cluster
- Service account
- Local configuration files

**Note**: GCS buckets and other persistent resources are not automatically deleted.

## Best Practices

1. **Use Confidential Computing**: Leverage hardware-based security for sensitive data
2. **Stop when not using**: Save costs by stopping the cluster
3. **Monitor costs**: Use Cloud Console billing dashboard
4. **Backup notebooks**: Save important work to cloud storage
5. **Use consistent credentials**: Keep storage credentials synchronized
6. **Test connectivity**: Verify storage access before processing data

## Next Steps

1. ✅ **Cluster created** - `altastata-confidential-cluster` (deleted to save costs)
2. ✅ **Jupyter Lab** - Successfully tested at `http://34.66.100.250:8888/lab`
3. ✅ **Multi-architecture support** - Works on both AMD64 (GCP) and ARM64 (Mac)
4. ✅ **AMD64 Architecture** - Confirmed running on AMD SEV confidential computing
5. ✅ **Cost optimization** - Cluster deleted to avoid charges
6. 🔄 **Recreate when needed** - Use `./setup-cluster.sh` to redeploy
7. 🔄 **Configure Altastata** - Set up your specific storage settings
8. 🔄 **Test Altastata operations** - File system and data processing

## Integration with Altastata

This setup is designed to work seamlessly with your Altastata Python package:

1. **Data Loading**: Altastata handles all data loading from cloud storage
2. **Model Training**: Altastata manages model and checkpoint storage
3. **Inference**: Altastata provides secure data processing
4. **Analytics**: Altastata handles all data analysis operations

## Support

For issues specific to:
- **GCP Confidential Computing**: [GCP Documentation](https://cloud.google.com/confidential-computing)
- **GKE**: [GKE Documentation](https://cloud.google.com/kubernetes-engine)
- **Altastata**: Check your Altastata package documentation

## License

This configuration is part of the Altastata project and follows the same licensing terms.

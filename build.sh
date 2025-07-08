#!/bin/bash

# Speaker Diarization API Build and Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="speaker-diarization-api"
TAG="latest"
REGISTRY="your-registry.com"  # Change this to your registry

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        log_warn "kubectl is not installed (needed for K8s deployment)"
    fi
    
    if ! nvidia-smi &> /dev/null; then
        log_warn "NVIDIA GPU not detected. Will use CPU mode."
    fi
    
    log_info "Requirements check completed"
}

build_image() {
    log_info "Building Docker image..."
    
    docker build -t ${IMAGE_NAME}:${TAG} .
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

test_image() {
    log_info "Testing Docker image..."
    
    # Start container in background
    CONTAINER_ID=$(docker run -d -p 8001:8000 ${IMAGE_NAME}:${TAG})
    
    # Wait for container to start
    sleep 30
    
    # Test health endpoint
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        log_info "Container health check passed"
    else
        log_error "Container health check failed"
        docker logs $CONTAINER_ID
        docker stop $CONTAINER_ID
        exit 1
    fi
    
    # Stop test container
    docker stop $CONTAINER_ID
    docker rm $CONTAINER_ID
    
    log_info "Image testing completed"
}

push_image() {
    if [ -n "$REGISTRY" ] && [ "$REGISTRY" != "your-registry.com" ]; then
        log_info "Pushing image to registry..."
        
        docker tag ${IMAGE_NAME}:${TAG} ${REGISTRY}/${IMAGE_NAME}:${TAG}
        docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}
        
        log_info "Image pushed to registry"
    else
        log_warn "Registry not configured, skipping push"
    fi
}

deploy_local() {
    log_info "Deploying locally with Docker Compose..."
    
    if [ ! -f ".env" ]; then
        log_info "Creating .env file from example..."
        cp .env.example .env
    fi
    
    docker-compose up -d
    
    log_info "Local deployment completed"
    log_info "API available at: http://localhost:8000"
    log_info "Docs available at: http://localhost:8000/docs"
}

deploy_k8s() {
    log_info "Deploying to Kubernetes..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required for Kubernetes deployment"
        exit 1
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s-deployment.yaml
    
    # Wait for deployment
    kubectl rollout status deployment/speaker-diarization-api
    
    log_info "Kubernetes deployment completed"
    
    # Get service info
    kubectl get services speaker-diarization-service
}

cleanup() {
    log_info "Cleaning up..."
    
    # Remove dangling images
    docker image prune -f
    
    log_info "Cleanup completed"
}

show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build          Build Docker image"
    echo "  test           Test Docker image"
    echo "  push           Push image to registry"
    echo "  deploy-local   Deploy locally with Docker Compose"
    echo "  deploy-k8s     Deploy to Kubernetes"
    echo "  full           Build, test, and deploy locally"
    echo "  cleanup        Clean up Docker resources"
    echo "  help           Show this help message"
}

# Main script
case "$1" in
    "build")
        check_requirements
        build_image
        ;;
    "test")
        test_image
        ;;
    "push")
        push_image
        ;;
    "deploy-local")
        deploy_local
        ;;
    "deploy-k8s")
        deploy_k8s
        ;;
    "full")
        check_requirements
        build_image
        test_image
        deploy_local
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        show_usage
        ;;
esac

log_info "Script completed!"

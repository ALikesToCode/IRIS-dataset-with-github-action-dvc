#!/bin/bash

# DVC Pipeline Execution Script
# Author: Abhyudaya B Tharakan 22f3001492

set -e

echo "=========================================="
echo "Running Iris Classification DVC Pipeline"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Check if DVC is set up
if [ ! -d ".dvc" ]; then
    echo "DVC not initialized. Running setup first..."
    ./setup_dvc.sh
fi

# Validate setup
print_step "Validating environment setup..."
python3 validate_setup.py

# Run DVC pipeline
print_step "Executing DVC pipeline..."
print_status "Running: dvc repro"
dvc repro

# Show results
print_step "Pipeline Results"
echo "Pipeline DAG:"
dvc dag

echo ""
echo "Metrics:"
dvc metrics show

# Push to remote if requested
if [ "$1" = "--push" ]; then
    print_step "Pushing data and artifacts to remote storage..."
    dvc push
    print_status "Data pushed to remote storage"
fi

echo ""
echo "=========================================="
echo "Pipeline execution completed successfully!"
echo "=========================================="
echo ""
echo "To view detailed logs: less iris_pipeline.log"
echo "To push data to remote: dvc push"
echo "To check status: dvc status" 
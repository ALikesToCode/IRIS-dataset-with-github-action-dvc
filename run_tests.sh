#!/bin/bash

# Local Test Runner for ML Pipeline
# This script runs the same tests that GitHub Actions will run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "test_pipeline.py" ]; then
    echo_error "test_pipeline.py not found. Run this script from the week-4 directory."
    exit 1
fi

echo_status "Starting ML Pipeline Tests..."
echo

# Install test dependencies if needed
echo_status "Checking test dependencies..."
if ! python -c "import pytest" 2>/dev/null; then
    echo_warning "Installing pytest..."
    pip install pytest pytest-cov
fi

# Check if data file exists
echo_status "Validating data..."
if [ -f "data/iris.csv" ]; then
    rows=$(python -c "import pandas as pd; print(len(pd.read_csv('data/iris.csv')))" 2>/dev/null || echo "unknown")
    echo_success "Data file exists: data/iris.csv ($rows rows)"
else
    echo_error "Data file missing: data/iris.csv"
    exit 1
fi

# Run unit tests
echo
echo_status "Running unit tests..."
if python -m pytest test_pipeline.py -v --tb=short; then
    echo_success "All unit tests passed!"
else
    echo_error "Some unit tests failed!"
    exit 1
fi

# Run integration test (model training)
echo
echo_status "Running model training validation..."
if python -c "
from main import Config, IrisPipeline
import os

config = Config(data_path='data/iris.csv')
pipeline = IrisPipeline(config)
model_path = pipeline.run_training_pipeline()

if os.path.exists(model_path):
    print('Model training completed successfully')
    print(f'Model saved to: {model_path}')
else:
    raise Exception('Model file not created')
"; then
    echo_success "Model training validation passed!"
else
    echo_error "Model training validation failed!"
    exit 1
fi

# Check model metrics if available
echo
echo_status "Checking model performance..."
if [ -f "artifacts/metrics.json" ]; then
    echo_success "Performance metrics:"
    cat artifacts/metrics.json | python -m json.tool
else
    echo_warning "No metrics file found"
fi

echo
echo_success "All tests completed successfully! 🎉"
echo
echo "Summary:"
echo "- ✅ Unit tests passed"
echo "- ✅ Model training validated"
echo "- ✅ Data quality checked"
echo
echo "You can now commit your changes and push to trigger GitHub Actions." 
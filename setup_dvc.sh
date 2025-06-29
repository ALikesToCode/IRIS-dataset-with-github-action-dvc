#!/bin/bash

# DVC Setup Script for Iris Classification Pipeline
# Author: Abhyudaya B Tharakan 22f3001492

set -e  # Exit on any error

echo "=========================================="
echo "DVC Setup for Iris Classification Pipeline"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "main.py" ] || [ ! -f "dvc_pipeline.py" ]; then
    print_error "Please run this script from the week-2 directory containing main.py and dvc_pipeline.py"
    exit 1
fi

# Step 1: Check if Git is initialized
print_step "Checking Git repository..."
if [ ! -d ".git" ]; then
    print_warning "Git repository not initialized. Initializing..."
    git init
    git add .
    git commit -m "Initial commit before DVC setup"
    print_status "Git repository initialized"
else
    print_status "Git repository already exists"
fi

# Step 2: Initialize DVC
print_step "Initializing DVC..."
if [ ! -d ".dvc" ]; then
    dvc init
    print_status "DVC initialized successfully"
else
    print_status "DVC already initialized"
fi

# Step 3: Configure DVC remote storage
print_step "Configuring DVC remote storage..."

# Read bucket URI from config or use default
BUCKET_URI=${GCS_BUCKET_URI:-"gs://iitm-mlops-week-1"}
REMOTE_URL="${BUCKET_URI}/dvc-storage"

print_status "Setting up remote storage: $REMOTE_URL"

# Remove existing remote if it exists and add new one
dvc remote remove gcs-storage 2>/dev/null || true
dvc remote add -d gcs-storage "$REMOTE_URL"

print_status "DVC remote storage configured"

# Step 4: Set up .gitignore for DVC
print_step "Setting up .gitignore..."
cat >> .gitignore << EOF

# DVC
/artifacts/model.joblib
/data/prepared/
*.dvc

# Logs
*.log
iris_pipeline.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

print_status ".gitignore updated"

# Step 5: Add data to DVC tracking
print_step "Adding data to DVC tracking..."
if [ -f "data/iris.csv" ]; then
    dvc add data/iris.csv
    print_status "Data file added to DVC tracking"
else
    print_error "data/iris.csv not found. Please ensure the data file exists."
    exit 1
fi

# Step 6: Create initial DVC pipeline
print_step "Setting up DVC pipeline..."
python3 -c "
from dvc_pipeline import DVCIrisPipeline, Config
config = Config.from_env()
pipeline = DVCIrisPipeline(config)
pipeline.setup_dvc_environment()
print('DVC pipeline setup completed')
"

print_status "DVC pipeline configuration created"

# Step 7: Commit DVC files to Git
print_step "Committing DVC configuration to Git..."
git add .dvc/ .dvcignore .gitignore data/.gitignore dvc.yaml params.yaml
git commit -m "Setup DVC pipeline with remote storage and data tracking"
print_status "DVC configuration committed to Git"

# Step 8: Run validation
print_step "Running setup validation..."
python3 validate_setup.py

# Step 9: Display next steps
echo ""
echo "=========================================="
echo "DVC Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run the DVC pipeline:"
echo "   python dvc_pipeline.py"
echo "   OR"
echo "   dvc repro"
echo ""
echo "2. Push data to remote storage:"
echo "   dvc push"
echo ""
echo "3. View pipeline DAG:"
echo "   dvc dag"
echo ""
echo "4. Check metrics:"
echo "   dvc metrics show"
echo ""
echo "5. For help with DVC commands:"
echo "   python validate_setup.py"
echo ""

print_status "DVC setup completed successfully!" 
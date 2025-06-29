# Git Workflow Guide for Week-4 MLOps Assignment

This guide covers the complete Git workflow for converting the week-4 codebase into a GitHub repository with automated testing and CI/CD.

## Part A: Convert Week 2's Codebase into a GitHub Repo

### Step 1: Check Local Repository Status

```bash
# Check current status
git status

# Verify latest commits
git log --oneline -5

# Check configured user
git config user.name
git config user.email
```

### Step 2: Create New GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon and select "New repository"
3. Name it `mlops-week4` (or your preferred name)
4. Make it public
5. Do NOT initialize with README (we already have content)
6. Copy the repository URL (e.g., `https://github.com/username/mlops-week4.git`)

### Step 3: Configure Remote Repository

```bash
# Remove existing origin if any
git remote remove origin

# Add your new GitHub repository as origin
git remote add origin https://github.com/YOUR_USERNAME/mlops-week4.git

# Verify remote configuration
git remote -v
```

### Step 4: Configure GitHub Authentication

#### Option A: Personal Access Token (Recommended)
1. Go to GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name like "MLOps Week 4"
4. Select scopes: `repo`, `workflow`, `write:packages`
5. Copy the generated token
6. Use this token instead of password when pushing

### Step 5: Push Code to Repository

```bash
# Push to main branch
git push -u origin main

# If you get an error about main vs master:
git branch -M main
git push -u origin main
```

## Part B: Write Unit Tests for Automatic Sanity Testing

### Running Tests Locally

```bash
# Make test runner executable
chmod +x run_tests.sh

# Run all tests
./run_tests.sh

# Or run pytest directly
python -m pytest test_pipeline.py -v
```

### Handling Detached HEAD State

If you encounter a detached HEAD state:

```bash
# Create and switch to a new branch
git checkout -b unit-tested-branch

# Push the new branch
git push -u origin unit-tested-branch

# Merge back to main
git checkout main
git pull origin main
git merge unit-tested-branch
git push origin main

# Clean up
git branch -d unit-tested-branch
git push origin --delete unit-tested-branch
```

## Part C: Configure GitHub Actions for Automated Testing

### Manual Workflow Execution

1. Go to your GitHub repository
2. Click "Actions" tab
3. Select "ML Pipeline Tests with CML Report" workflow
4. Click "Run workflow" button
5. Choose branch (usually `main`)
6. Click "Run workflow"

The workflow will:
- Run unit tests with coverage
- Validate model training
- Check data quality
- Post results as comments using CML 
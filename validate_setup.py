#!/usr/bin/env python3
"""
Setup Validation Script for Iris Classification Pipeline with DVC

This script validates that all required dependencies and configurations
are properly set up before running the main pipeline.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"


def check_required_packages() -> List[Tuple[bool, str]]:
    """Check if required Python packages are installed."""
    required_packages = [
        'pandas',
        'numpy', 
        'sklearn',
        'joblib',
        'google.cloud.aiplatform',
        'dvc',
        'yaml'
    ]
    
    results = []
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            else:
                __import__(package)
            results.append((True, f"✓ {package}"))
        except ImportError:
            results.append((False, f"✗ {package} (not installed)"))
    
    return results


def check_dvc_installation() -> Tuple[bool, str]:
    """Check if DVC is properly installed."""
    try:
        result = subprocess.run(['dvc', 'version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # Extract version from output
            version_line = result.stdout.split('\n')[0]
            return True, f"✓ DVC ({version_line})"
        else:
            return False, "✗ DVC (not working properly)"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "✗ DVC (not installed)"


def check_dvc_git_integration() -> Tuple[bool, str]:
    """Check if Git is available for DVC integration."""
    try:
        result = subprocess.run(['git', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, "✓ Git (required for DVC)"
        else:
            return False, "✗ Git (not working)"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "✗ Git (not installed - required for DVC)"


def check_gcloud_cli() -> Tuple[bool, str]:
    """Check if gcloud CLI is installed and configured."""
    try:
        result = subprocess.run(['gcloud', 'version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, "✓ gcloud CLI"
        else:
            return False, "✗ gcloud CLI (not working)"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "✗ gcloud CLI (not installed)"


def check_gsutil() -> Tuple[bool, str]:
    """Check if gsutil is available."""
    try:
        result = subprocess.run(['gsutil', 'version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, "✓ gsutil"
        else:
            return False, "✗ gsutil (not working)"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "✗ gsutil (not installed)"


def check_authentication() -> Tuple[bool, str]:
    """Check if Google Cloud authentication is set up."""
    try:
        result = subprocess.run(['gcloud', 'auth', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'ACTIVE' in result.stdout:
            return True, "✓ Google Cloud authentication"
        else:
            return False, "✗ Google Cloud authentication (not configured)"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "✗ Google Cloud authentication (gcloud not available)"


def check_data_file() -> Tuple[bool, str]:
    """Check if the iris.csv data file exists."""
    data_path = Path("data/iris.csv")
    if data_path.exists():
        return True, f"✓ Data file ({data_path})"
    else:
        return False, f"✗ Data file missing ({data_path})"


def check_directory_structure() -> List[Tuple[bool, str]]:
    """Check if required directories exist."""
    required_dirs = ['data', 'artifacts']
    results = []
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            results.append((True, f"✓ Directory: {dir_name}/"))
        else:
            # Try to create the directory
            try:
                dir_path.mkdir(exist_ok=True)
                results.append((True, f"✓ Directory: {dir_name}/ (created)"))
            except Exception as e:
                results.append((False, f"✗ Directory: {dir_name}/ (cannot create: {e})"))
    
    return results


def check_dvc_setup() -> List[Tuple[bool, str]]:
    """Check DVC-specific setup requirements."""
    results = []
    
    # Check if DVC is initialized
    if Path(".dvc").exists():
        results.append((True, "✓ DVC repository (initialized)"))
        
        # Check DVC config
        dvc_config = Path(".dvc/config")
        if dvc_config.exists():
            results.append((True, "✓ DVC configuration file"))
        else:
            results.append((False, "✗ DVC configuration file (missing)"))
    else:
        results.append((False, "✗ DVC repository (not initialized)"))
    
    # Check if dvcignore exists
    if Path(".dvcignore").exists():
        results.append((True, "✓ .dvcignore file"))
    else:
        results.append((False, "✗ .dvcignore file (missing)"))
    
    return results


def print_results(title: str, results: List[Tuple[bool, str]]) -> int:
    """Print validation results and return number of failures."""
    print(f"\n{title}:")
    print("-" * len(title))
    
    failures = 0
    for success, message in results:
        print(f"  {message}")
        if not success:
            failures += 1
    
    return failures


def print_dvc_commands_help():
    """Print helpful DVC commands for setup."""
    print("\n" + "="*60)
    print("DVC SETUP COMMANDS REFERENCE")
    print("="*60)
    print("\n🔧 Basic DVC Setup:")
    print("  dvc init                    # Initialize DVC in current directory")
    print("  dvc remote add -d gcs gs://bucket/path  # Add Google Cloud Storage remote")
    print("  dvc add data/               # Start tracking data folder")
    print("  dvc push                    # Push data to remote storage")
    print("  dvc pull                    # Pull data from remote storage")
    
    print("\n📊 Pipeline Commands:")
    print("  dvc repro                   # Run/reproduce the entire pipeline")
    print("  dvc repro stage_name        # Run specific pipeline stage")
    print("  dvc dag                     # Show pipeline dependency graph")
    print("  dvc metrics show            # Show pipeline metrics")
    
    print("\n🔄 Data Management:")
    print("  dvc status                  # Check status of tracked files")
    print("  dvc diff                    # Show changes in data")
    print("  dvc checkout                # Restore tracked files to last committed state")
    
    print("\n💾 Version Control:")
    print("  git add .dvc/ dvc.yaml params.yaml data/.gitignore")
    print("  git commit -m 'Setup DVC pipeline'")
    print("  dvc push                    # Push data to remote after git commit")


def main():
    """Run all validation checks."""
    print("Iris Classification Pipeline - Setup Validation (with DVC)")
    print("=" * 60)
    
    total_failures = 0
    
    # Check Python version
    success, message = check_python_version()
    print(f"\nPython Version:")
    print("-" * 14)
    print(f"  {message}")
    if not success:
        total_failures += 1
    
    # Check required packages
    package_results = check_required_packages()
    total_failures += print_results("Required Python Packages", package_results)
    
    # Check DVC and Git
    dvc_success, dvc_message = check_dvc_installation()
    git_success, git_message = check_dvc_git_integration()
    dvc_tools_results = [(dvc_success, dvc_message), (git_success, git_message)]
    total_failures += print_results("DVC and Version Control Tools", dvc_tools_results)
    
    # Check directory structure
    dir_results = check_directory_structure()
    total_failures += print_results("Directory Structure", dir_results)
    
    # Check data file
    success, message = check_data_file()
    print(f"\nData File:")
    print("-" * 9)
    print(f"  {message}")
    if not success:
        total_failures += 1
    
    # Check DVC setup
    dvc_setup_results = check_dvc_setup()
    total_failures += print_results("DVC Setup", dvc_setup_results)
    
    # Check Google Cloud tools
    gcloud_success, gcloud_message = check_gcloud_cli()
    gsutil_success, gsutil_message = check_gsutil()
    auth_success, auth_message = check_authentication()
    
    gcloud_results = [(gcloud_success, gcloud_message),
                     (gsutil_success, gsutil_message),
                     (auth_success, auth_message)]
    total_failures += print_results("Google Cloud Tools", gcloud_results)
    
    # Summary
    print(f"\n{'='*60}")
    if total_failures == 0:
        print("🎉 All checks passed! Your environment is ready for DVC pipeline.")
        print("\nYou can now run the pipeline with:")
        print("  python dvc_pipeline.py      # Run DVC-managed pipeline")
        print("  python main.py              # Run basic pipeline")
        print("  dvc repro                   # Run DVC pipeline directly")
    else:
        print(f"❌ {total_failures} check(s) failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  • Install missing packages: pip install -r requirements.txt")
        print("  • Install DVC: pip install 'dvc[gs]'")
        print("  • Install Git: https://git-scm.com/downloads")
        print("  • Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
        print("  • Authenticate: gcloud auth login && gcloud auth application-default login")
        print("  • Initialize DVC: dvc init")
        print("  • Set up DVC remote: dvc remote add -d gcs gs://your-bucket/dvc-storage")
    
    # Always show DVC commands help
    print_dvc_commands_help()
    
    return total_failures


if __name__ == "__main__":
    exit_code = main()
    sys.exit(min(exit_code, 1))  # Cap at 1 for shell compatibility 
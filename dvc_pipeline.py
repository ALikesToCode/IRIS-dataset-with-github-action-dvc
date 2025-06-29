#!/usr/bin/env python3
"""
DVC-Enhanced Iris Classification Pipeline

This module extends the main pipeline with DVC integration for data version control
and reproducible ML pipeline execution.

Author: Abhyudaya B Tharakan 22f3001492
"""

import logging
import os
import sys
import subprocess
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from main import IrisPipeline, Config, IrisModelTrainer, VertexAIDeployer

# Configure logging
logger = logging.getLogger(__name__)


class DVCPipelineManager:
    """Manages DVC operations for the Iris classification pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.dvc_root = Path.cwd()
        self.pipeline_file = self.dvc_root / "dvc.yaml"
        self.params_file = self.dvc_root / "params.yaml"
        
    def initialize_dvc(self) -> bool:
        """Initialize DVC repository and configure remote storage."""
        try:
            # Check if DVC is already initialized
            if not (self.dvc_root / ".dvc").exists():
                logger.info("Initializing DVC repository...")
                result = subprocess.run(["dvc", "init"], 
                                      capture_output=True, text=True, cwd=self.dvc_root)
                if result.returncode != 0:
                    logger.error(f"Failed to initialize DVC: {result.stderr}")
                    return False
                logger.info("DVC repository initialized successfully")
            else:
                logger.info("DVC repository already initialized")
            
            # Configure remote storage
            remote_url = f"{self.config.bucket_uri}/dvc-storage"
            logger.info(f"Configuring DVC remote storage: {remote_url}")
            
            # Add remote
            result = subprocess.run([
                "dvc", "remote", "add", "-f", "gcs-storage", remote_url
            ], capture_output=True, text=True, cwd=self.dvc_root)
            
            if result.returncode != 0:
                logger.warning(f"Remote might already exist: {result.stderr}")
            
            # Set default remote
            result = subprocess.run([
                "dvc", "remote", "default", "gcs-storage"
            ], capture_output=True, text=True, cwd=self.dvc_root)
            
            if result.returncode != 0:
                logger.error(f"Failed to set default remote: {result.stderr}")
                return False
            
            logger.info("DVC remote storage configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing DVC: {e}")
            return False
    
    def create_params_file(self) -> bool:
        """Create DVC parameters file with pipeline configuration."""
        try:
            params = {
                "model": {
                    "max_depth": self.config.max_depth,
                    "random_state": self.config.random_state
                },
                "data": {
                    "test_size": self.config.test_size,
                    "data_path": self.config.data_path
                },
                "training": {
                    "artifacts_dir": self.config.artifacts_dir,
                    "model_filename": self.config.model_filename
                },
                "deployment": {
                    "project_id": self.config.project_id,
                    "location": self.config.location,
                    "bucket_uri": self.config.bucket_uri,
                    "model_artifact_dir": self.config.model_artifact_dir
                }
            }
            
            with open(self.params_file, 'w') as f:
                yaml.dump(params, f, default_flow_style=False, indent=2)
            
            logger.info(f"Parameters file created: {self.params_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating parameters file: {e}")
            return False
    
    def create_dvc_pipeline(self) -> bool:
        """Create DVC pipeline definition."""
        try:
            pipeline_config = {
                "stages": {
                    "prepare_data": {
                        "cmd": "python -c \"from dvc_pipeline import prepare_data_stage; prepare_data_stage()\"",
                        "deps": ["data/iris.csv", "dvc_pipeline.py"],
                        "params": ["data.test_size", "data.data_path"],
                        "outs": ["data/prepared"]
                    },
                    "train_model": {
                        "cmd": "python -c \"from dvc_pipeline import train_model_stage; train_model_stage()\"",
                        "deps": ["data/prepared", "dvc_pipeline.py"],
                        "params": ["model.max_depth", "model.random_state", "training"],
                        "outs": ["artifacts/model.joblib"],
                        "metrics": ["artifacts/metrics.json"]
                    },
                    "validate_model": {
                        "cmd": "python -c \"from dvc_pipeline import validate_model_stage; validate_model_stage()\"",
                        "deps": ["artifacts/model.joblib", "data/prepared", "dvc_pipeline.py"],
                        "metrics": ["artifacts/validation_metrics.json"]
                    },
                    "deploy_model": {
                        "cmd": "python -c \"from dvc_pipeline import deploy_model_stage; deploy_model_stage()\"",
                        "deps": ["artifacts/model.joblib", "dvc_pipeline.py"],
                        "params": ["deployment"],
                        "outs": ["artifacts/deployment_info.json"]
                    }
                }
            }
            
            with open(self.pipeline_file, 'w') as f:
                yaml.dump(pipeline_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"DVC pipeline created: {self.pipeline_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating DVC pipeline: {e}")
            return False
    
    def add_data_to_dvc(self) -> bool:
        """Add data files to DVC tracking."""
        try:
            data_path = Path(self.config.data_path)
            if not data_path.exists():
                logger.error(f"Data file not found: {data_path}")
                return False
            
            logger.info(f"Adding data file to DVC: {data_path}")
            result = subprocess.run([
                "dvc", "add", str(data_path)
            ], capture_output=True, text=True, cwd=self.dvc_root)
            
            if result.returncode != 0:
                logger.error(f"Failed to add data to DVC: {result.stderr}")
                return False
            
            logger.info("Data file added to DVC successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding data to DVC: {e}")
            return False
    
    def run_dvc_pipeline(self, stage: Optional[str] = None) -> bool:
        """Run the DVC pipeline."""
        try:
            cmd = ["dvc", "repro"]
            if stage:
                cmd.append(stage)
            
            logger.info(f"Running DVC pipeline: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.dvc_root)
            
            if result.returncode == 0:
                logger.info("DVC pipeline completed successfully")
                return True
            else:
                logger.error("DVC pipeline failed")
                return False
                
        except Exception as e:
            logger.error(f"Error running DVC pipeline: {e}")
            return False
    
    def push_to_remote(self) -> bool:
        """Push data and pipeline artifacts to DVC remote storage."""
        try:
            logger.info("Pushing data to DVC remote storage...")
            result = subprocess.run(["dvc", "push"], 
                                  capture_output=True, text=True, cwd=self.dvc_root)
            
            if result.returncode == 0:
                logger.info("Data pushed to remote storage successfully")
                return True
            else:
                logger.error(f"Failed to push to remote: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error pushing to remote: {e}")
            return False
    
    def pull_from_remote(self) -> bool:
        """Pull data and pipeline artifacts from DVC remote storage."""
        try:
            logger.info("Pulling data from DVC remote storage...")
            result = subprocess.run(["dvc", "pull"], 
                                  capture_output=True, text=True, cwd=self.dvc_root)
            
            if result.returncode == 0:
                logger.info("Data pulled from remote storage successfully")
                return True
            else:
                logger.error(f"Failed to pull from remote: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error pulling from remote: {e}")
            return False


class DVCIrisPipeline(IrisPipeline):
    """DVC-enhanced Iris pipeline that extends the base pipeline."""
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.dvc_manager = DVCPipelineManager(self.config)
    
    def setup_dvc_environment(self) -> bool:
        """Set up the complete DVC environment."""
        logger.info("Setting up DVC environment...")
        
        try:
            # Initialize DVC
            if not self.dvc_manager.initialize_dvc():
                return False
            
            # Create parameters file
            if not self.dvc_manager.create_params_file():
                return False
            
            # Create DVC pipeline
            if not self.dvc_manager.create_dvc_pipeline():
                return False
            
            # Add data to DVC
            if not self.dvc_manager.add_data_to_dvc():
                return False
            
            logger.info("DVC environment setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup DVC environment: {e}")
            return False
    
    def run_dvc_pipeline(self) -> bool:
        """Run the complete DVC pipeline."""
        logger.info("Starting DVC-managed pipeline execution...")
        
        try:
            # Setup DVC environment if not already done
            if not Path(".dvc").exists():
                if not self.setup_dvc_environment():
                    return False
            
            # Run the DVC pipeline
            if not self.dvc_manager.run_dvc_pipeline():
                return False
            
            # Push results to remote storage
            if not self.dvc_manager.push_to_remote():
                logger.warning("Failed to push to remote storage, but pipeline completed")
            
            logger.info("DVC pipeline execution completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"DVC pipeline execution failed: {e}")
            return False


# DVC Stage Functions
def prepare_data_stage():
    """DVC stage: Prepare and validate data."""
    import pandas as pd
    from pathlib import Path
    import yaml
    
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Load and prepare data
    data_path = params['data']['data_path']
    data = pd.read_csv(data_path)
    
    # Create prepared data directory
    prepared_dir = Path('data/prepared')
    prepared_dir.mkdir(exist_ok=True)
    
    # Save prepared data
    data.to_csv(prepared_dir / 'iris_prepared.csv', index=False)
    
    # Save data info
    data_info = {
        'shape': data.shape,
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict()
    }
    
    with open(prepared_dir / 'data_info.json', 'w') as f:
        json.dump(data_info, f, indent=2, default=str)
    
    logger.info(f"Data preparation completed. Shape: {data.shape}")


def train_model_stage():
    """DVC stage: Train the model."""
    import yaml
    from main import Config, IrisModelTrainer
    
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Create config from parameters
    config = Config(
        max_depth=params['model']['max_depth'],
        random_state=params['model']['random_state'],
        test_size=params['data']['test_size'],
        artifacts_dir=params['training']['artifacts_dir'],
        model_filename=params['training']['model_filename']
    )
    
    # Train model
    trainer = IrisModelTrainer(config)
    data = trainer.load_data()
    X_train, y_train, X_test, y_test = trainer.prepare_data(data)
    trainer.train_model(X_train, y_train)
    metrics = trainer.evaluate_model(X_test, y_test)
    trainer.save_model()
    
    logger.info("Model training stage completed")


def validate_model_stage():
    """DVC stage: Validate the trained model."""
    import joblib
    import pandas as pd
    import json
    from sklearn import metrics
    from pathlib import Path
    
    # Load model and validation data
    model = joblib.load('artifacts/model.joblib')
    data = pd.read_csv('data/prepared/iris_prepared.csv')
    
    # Prepare validation data (use a different split for validation)
    from sklearn.model_selection import train_test_split
    
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = data[feature_columns]
    y = data['species']
    
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    
    # Validate model
    predictions = model.predict(X_val)
    
    validation_metrics = {
        'validation_accuracy': float(metrics.accuracy_score(y_val, predictions)),
        'validation_precision': float(metrics.precision_score(y_val, predictions, average='weighted')),
        'validation_recall': float(metrics.recall_score(y_val, predictions, average='weighted')),
        'validation_f1': float(metrics.f1_score(y_val, predictions, average='weighted')),
        'validation_samples': len(y_val)
    }
    
    # Save validation metrics
    with open('artifacts/validation_metrics.json', 'w') as f:
        json.dump(validation_metrics, f, indent=2)
    
    logger.info("Model validation stage completed")


def deploy_model_stage():
    """DVC stage: Deploy the model."""
    import yaml
    from main import Config, VertexAIDeployer
    
    # Load parameters
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Create config from parameters
    config = Config(
        project_id=params['deployment']['project_id'],
        location=params['deployment']['location'],
        bucket_uri=params['deployment']['bucket_uri'],
        model_artifact_dir=params['deployment']['model_artifact_dir']
    )
    
    # Deploy model
    deployer = VertexAIDeployer(config)
    
    # Create bucket and upload artifacts
    deployer.create_bucket_if_not_exists()
    success = deployer.upload_model_artifacts('artifacts/model.joblib')
    
    # Save deployment info
    deployment_info = {
        'deployment_successful': success,
        'bucket_uri': config.bucket_uri,
        'model_artifact_dir': config.model_artifact_dir,
        'project_id': config.project_id,
        'location': config.location
    }
    
    with open('artifacts/deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    logger.info("Model deployment stage completed")


def main():
    """Main entry point for DVC pipeline."""
    try:
        # Load configuration
        config = Config.from_env()
        
        # Create and run DVC pipeline
        pipeline = DVCIrisPipeline(config)
        success = pipeline.run_dvc_pipeline()
        
        if success:
            logger.info("DVC Pipeline execution completed successfully!")
            sys.exit(0)
        else:
            logger.error("DVC Pipeline execution failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("DVC Pipeline execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
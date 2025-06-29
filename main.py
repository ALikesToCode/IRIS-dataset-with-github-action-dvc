#!/usr/bin/env python3
"""
Iris Classification Model Training and Deployment Pipeline

This module provides a complete pipeline for training an Iris classification model
using scikit-learn and deploying it to Google Vertex AI using custom containers.

Author: Abhyudaya B Tharakan 22f3001492
"""

import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import json

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from google.cloud import aiplatform
from google.cloud.exceptions import GoogleCloudError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('iris_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration class for the Iris classification pipeline."""
    
    # Google Cloud Configuration
    project_id: str = "steady-triumph-447006-f8"
    location: str = "asia-south1"
    bucket_uri: str = "gs://iitm-mlops-week-1"
    
    # Model Configuration
    model_artifact_dir: str = "my-models/iris-classifier-week-1"
    repository: str = "iris-classifier-repo"
    image: str = "iris-classifier-img"
    model_display_name: str = "iris-classifier"
    
    # Training Configuration
    test_size: float = 0.4
    random_state: int = 42
    max_depth: int = 3
    
    # File paths
    data_path: str = "data/iris.csv"
    artifacts_dir: str = "artifacts"
    model_filename: str = "model.joblib"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            project_id=os.getenv('GCP_PROJECT_ID', cls.project_id),
            location=os.getenv('GCP_LOCATION', cls.location),
            bucket_uri=os.getenv('GCS_BUCKET_URI', cls.bucket_uri),
            model_artifact_dir=os.getenv('MODEL_ARTIFACT_DIR', cls.model_artifact_dir),
            repository=os.getenv('ARTIFACT_REPOSITORY', cls.repository),
            image=os.getenv('CONTAINER_IMAGE', cls.image),
            model_display_name=os.getenv('MODEL_DISPLAY_NAME', cls.model_display_name),
        )


class IrisModelTrainer:
    """Handles training of the Iris classification model."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[DecisionTreeClassifier] = None
        self.model_metrics: Dict[str, float] = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load and validate the Iris dataset."""
        try:
            if not Path(self.config.data_path).exists():
                raise FileNotFoundError(f"Data file not found: {self.config.data_path}")
            
            data = pd.read_csv(self.config.data_path)
            logger.info(f"Loaded dataset with shape: {data.shape}")
            
            # Validate required columns
            required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Check for missing values
            if data.isnull().sum().sum() > 0:
                logger.warning("Dataset contains missing values. Consider data cleaning.")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Split and prepare the data for training."""
        try:
            # Features and target
            feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            X = data[feature_columns]
            y = data['species']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                stratify=y,
                random_state=self.config.random_state
            )
            
            logger.info(f"Training set size: {X_train.shape[0]}")
            logger.info(f"Test set size: {X_test.shape[0]}")
            logger.info(f"Target classes: {y.unique().tolist()}")
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
        """Train the Decision Tree classifier."""
        try:
            self.model = DecisionTreeClassifier(
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
            
            logger.info("Training Decision Tree classifier...")
            self.model.fit(X_train, y_train)
            logger.info("Model training completed successfully")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the trained model and return metrics."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        try:
            predictions = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = metrics.accuracy_score(y_test, predictions)
            precision = metrics.precision_score(y_test, predictions, average='weighted')
            recall = metrics.recall_score(y_test, predictions, average='weighted')
            f1 = metrics.f1_score(y_test, predictions, average='weighted')
            
            self.model_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            logger.info(f"Model Evaluation Results:")
            for metric, value in self.model_metrics.items():
                logger.info(f"  {metric.capitalize()}: {value:.4f}")
            
            return self.model_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def save_model(self) -> str:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        try:
            # Ensure artifacts directory exists
            artifacts_path = Path(self.config.artifacts_dir)
            artifacts_path.mkdir(exist_ok=True)
            
            model_path = artifacts_path / self.config.model_filename
            joblib.dump(self.model, model_path)
            
            # Save metrics
            metrics_path = artifacts_path / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.model_metrics, f, indent=2)
            
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Metrics saved to: {metrics_path}")
            
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise


class VertexAIDeployer:
    """Handles deployment to Google Vertex AI."""
    
    def __init__(self, config: Config):
        self.config = config
        self._initialize_vertex_ai()
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI client."""
        try:
            aiplatform.init(
                project=self.config.project_id,
                location=self.config.location,
                staging_bucket=self.config.bucket_uri
            )
            logger.info("Vertex AI client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Vertex AI: {e}")
            raise
    
    def create_bucket_if_not_exists(self) -> bool:
        """Create GCS bucket if it doesn't exist."""
        try:
            # Check if bucket exists by trying to list it
            cmd = ["gsutil", "ls", self.config.bucket_uri]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Bucket {self.config.bucket_uri} already exists")
                return True
            
            # Create bucket
            cmd = [
                "gsutil", "mb",
                "-l", self.config.location,
                "-p", self.config.project_id,
                self.config.bucket_uri
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Created bucket: {self.config.bucket_uri}")
                return True
            else:
                logger.error(f"Error creating bucket: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking/creating bucket: {e}")
            return False
    
    def upload_model_artifacts(self, model_path: str) -> bool:
        """Upload model artifacts to GCS."""
        try:
            gcs_destination = f"{self.config.bucket_uri}/{self.config.model_artifact_dir}/"
            
            cmd = ["gsutil", "cp", model_path, gcs_destination]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Model artifacts uploaded to: {gcs_destination}")
                return True
            else:
                logger.error(f"Error uploading artifacts: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error uploading model artifacts: {e}")
            return False


class IrisPipeline:
    """Main pipeline class that orchestrates the entire process."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.from_env()
        self.trainer = IrisModelTrainer(self.config)
        self.deployer = VertexAIDeployer(self.config)
    
    def run_training_pipeline(self) -> str:
        """Run the complete training pipeline."""
        logger.info("Starting Iris classification training pipeline...")
        
        try:
            # Load and prepare data
            data = self.trainer.load_data()
            X_train, y_train, X_test, y_test = self.trainer.prepare_data(data)
            
            # Train model
            self.trainer.train_model(X_train, y_train)
            
            # Evaluate model
            metrics = self.trainer.evaluate_model(X_test, y_test)
            
            # Save model
            model_path = self.trainer.save_model()
            
            logger.info("Training pipeline completed successfully!")
            return model_path
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def run_deployment_pipeline(self, model_path: str) -> bool:
        """Run the deployment pipeline."""
        logger.info("Starting deployment pipeline...")
        
        try:
            # Create bucket if needed
            if not self.deployer.create_bucket_if_not_exists():
                raise RuntimeError("Failed to create or access GCS bucket")
            
            # Upload model artifacts
            if not self.deployer.upload_model_artifacts(model_path):
                raise RuntimeError("Failed to upload model artifacts")
            
            logger.info("Deployment pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Deployment pipeline failed: {e}")
            raise
    
    def run_full_pipeline(self) -> bool:
        """Run both training and deployment pipelines."""
        try:
            model_path = self.run_training_pipeline()
            return self.run_deployment_pipeline(model_path)
        except Exception as e:
            logger.error(f"Full pipeline failed: {e}")
            return False


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = Config.from_env()
        
        # Create and run pipeline
        pipeline = IrisPipeline(config)
        success = pipeline.run_full_pipeline()
        
        if success:
            logger.info("Pipeline execution completed successfully!")
            sys.exit(0)
        else:
            logger.error("Pipeline execution failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()



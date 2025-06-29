#!/usr/bin/env python3
"""
Unit Tests for Iris Classification Pipeline

This module contains comprehensive unit tests for the Iris classification pipeline,
including data validation, model training, evaluation, and deployment components.

Author: Abhyudaya B Tharakan 22f3001492
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import Config, IrisModelTrainer, VertexAIDeployer, IrisPipeline
from dvc_pipeline import DVCPipelineManager, DVCIrisPipeline


class TestConfig:
    """Test cases for Configuration management."""
    
    def test_default_config_creation(self):
        """Test creation of default configuration."""
        config = Config()
        assert config.project_id == "steady-triumph-447006-f8"
        assert config.location == "asia-south1"
        assert config.test_size == 0.4
        assert config.random_state == 42
        assert config.max_depth == 3
        
    def test_config_from_env(self):
        """Test configuration creation from environment variables."""
        with patch.dict(os.environ, {
            'GCP_PROJECT_ID': 'test-project',
            'GCP_LOCATION': 'us-central1',
            'GCS_BUCKET_URI': 'gs://test-bucket'
        }):
            config = Config.from_env()
            assert config.project_id == 'test-project'
            assert config.location == 'us-central1'
            assert config.bucket_uri == 'gs://test-bucket'


class TestIrisModelTrainer:
    """Test cases for IrisModelTrainer class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config(
            data_path="test_data.csv",
            artifacts_dir="test_artifacts",
            test_size=0.3,
            random_state=42,
            max_depth=3
        )
    
    @pytest.fixture
    def trainer(self, config):
        """Create a trainer instance."""
        return IrisModelTrainer(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample iris data for testing."""
        np.random.seed(42)
        data = {
            'sepal_length': np.random.normal(5.8, 0.8, 150),
            'sepal_width': np.random.normal(3.0, 0.4, 150),
            'petal_length': np.random.normal(3.8, 1.8, 150),
            'petal_width': np.random.normal(1.2, 0.8, 150),
            'species': ['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_data_file(self, sample_data, tmp_path):
        """Create a temporary data file."""
        data_file = tmp_path / "test_data.csv"
        sample_data.to_csv(data_file, index=False)
        return str(data_file)
    
    def test_load_data_success(self, trainer, temp_data_file, config):
        """Test successful data loading."""
        config.data_path = temp_data_file
        trainer.config = config
        
        data = trainer.load_data()
        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] == 150
        assert set(data.columns) == {'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'}
    
    def test_load_data_file_not_found(self, trainer):
        """Test data loading when file doesn't exist."""
        trainer.config.data_path = "nonexistent_file.csv"
        with pytest.raises(FileNotFoundError):
            trainer.load_data()
    
    def test_load_data_missing_columns(self, trainer, tmp_path):
        """Test data loading with missing required columns."""
        # Create data with missing columns
        incomplete_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9],
            'sepal_width': [3.5, 3.0],
            'species': ['setosa', 'setosa']
        })
        
        data_file = tmp_path / "incomplete_data.csv"
        incomplete_data.to_csv(data_file, index=False)
        trainer.config.data_path = str(data_file)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            trainer.load_data()
    
    def test_prepare_data(self, trainer, sample_data):
        """Test data preparation and splitting."""
        X_train, y_train, X_test, y_test = trainer.prepare_data(sample_data)
        
        # Check shapes
        total_samples = len(sample_data)
        expected_test_size = int(total_samples * trainer.config.test_size)
        expected_train_size = total_samples - expected_test_size
        
        assert X_train.shape[0] == expected_train_size
        assert X_test.shape[0] == expected_test_size
        assert len(y_train) == expected_train_size
        assert len(y_test) == expected_test_size
        
        # Check feature columns
        expected_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        assert list(X_train.columns) == expected_features
        assert list(X_test.columns) == expected_features
        
        # Check species distribution
        assert set(y_train.unique()).issubset({'setosa', 'versicolor', 'virginica'})
        assert set(y_test.unique()).issubset({'setosa', 'versicolor', 'virginica'})
    
    def test_train_model(self, trainer, sample_data):
        """Test model training."""
        X_train, y_train, _, _ = trainer.prepare_data(sample_data)
        model = trainer.train_model(X_train, y_train)
        
        assert trainer.model is not None
        assert hasattr(trainer.model, 'predict')
        assert hasattr(trainer.model, 'fit')
        assert trainer.model.max_depth == trainer.config.max_depth
        assert trainer.model.random_state == trainer.config.random_state
    
    def test_evaluate_model(self, trainer, sample_data):
        """Test model evaluation."""
        X_train, y_train, X_test, y_test = trainer.prepare_data(sample_data)
        trainer.train_model(X_train, y_train)
        
        metrics = trainer.evaluate_model(X_test, y_test)
        
        assert isinstance(metrics, dict)
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        assert all(metric in metrics for metric in expected_metrics)
        assert all(0 <= metrics[metric] <= 1 for metric in expected_metrics)
    
    def test_evaluate_model_not_trained(self, trainer):
        """Test evaluation when model is not trained."""
        with pytest.raises(ValueError, match="Model not trained yet"):
            trainer.evaluate_model(pd.DataFrame(), pd.Series())
    
    def test_save_model(self, trainer, sample_data, tmp_path):
        """Test model saving."""
        trainer.config.artifacts_dir = str(tmp_path / "artifacts")
        
        X_train, y_train, X_test, y_test = trainer.prepare_data(sample_data)
        trainer.train_model(X_train, y_train)
        trainer.evaluate_model(X_test, y_test)
        
        model_path = trainer.save_model()
        
        # Check if files are created
        artifacts_dir = Path(trainer.config.artifacts_dir)
        model_file = artifacts_dir / trainer.config.model_filename
        metrics_file = artifacts_dir / "metrics.json"
        
        assert model_file.exists()
        assert metrics_file.exists()
        assert str(model_file) == model_path
        
        # Check if model can be loaded
        loaded_model = joblib.load(model_file)
        assert hasattr(loaded_model, 'predict')
        
        # Check metrics file content
        with open(metrics_file, 'r') as f:
            saved_metrics = json.load(f)
        assert isinstance(saved_metrics, dict)
        assert 'accuracy' in saved_metrics
    
    def test_save_model_not_trained(self, trainer):
        """Test saving when model is not trained."""
        with pytest.raises(ValueError, match="Model not trained yet"):
            trainer.save_model()


class TestVertexAIDeployer:
    """Test cases for VertexAIDeployer class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config(
            project_id="test-project",
            location="us-central1",
            bucket_uri="gs://test-bucket"
        )
    
    @patch('main.aiplatform.init')
    def test_deployer_initialization(self, mock_init, config):
        """Test deployer initialization."""
        deployer = VertexAIDeployer(config)
        mock_init.assert_called_once_with(
            project=config.project_id,
            location=config.location,
            staging_bucket=config.bucket_uri
        )
    
    @patch('main.subprocess.run')
    def test_create_bucket_exists(self, mock_run, config):
        """Test bucket creation when bucket already exists."""
        # Mock successful gsutil ls (bucket exists)
        mock_run.return_value.returncode = 0
        
        deployer = VertexAIDeployer(config)
        result = deployer.create_bucket_if_not_exists()
        
        assert result is True
        mock_run.assert_called_once()
    
    @patch('main.subprocess.run')
    def test_create_bucket_new(self, mock_run, config):
        """Test creating a new bucket."""
        # Mock gsutil ls failure (bucket doesn't exist) then successful creation
        mock_run.side_effect = [
            Mock(returncode=1),  # ls fails
            Mock(returncode=0)   # mb succeeds
        ]
        
        deployer = VertexAIDeployer(config)
        result = deployer.create_bucket_if_not_exists()
        
        assert result is True
        assert mock_run.call_count == 2
    
    @patch('main.subprocess.run')
    def test_upload_model_artifacts_success(self, mock_run, config):
        """Test successful model artifact upload."""
        mock_run.return_value.returncode = 0
        
        deployer = VertexAIDeployer(config)
        result = deployer.upload_model_artifacts("test_model.joblib")
        
        assert result is True
        mock_run.assert_called_once()
    
    @patch('main.subprocess.run')
    def test_upload_model_artifacts_failure(self, mock_run, config):
        """Test failed model artifact upload."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Upload failed"
        
        deployer = VertexAIDeployer(config)
        result = deployer.upload_model_artifacts("test_model.joblib")
        
        assert result is False


class TestIrisPipeline:
    """Test cases for the complete IrisPipeline."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config(
            data_path="test_data.csv",
            artifacts_dir="test_artifacts"
        )
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample data file."""
        np.random.seed(42)
        data = {
            'sepal_length': np.random.normal(5.8, 0.8, 150),
            'sepal_width': np.random.normal(3.0, 0.4, 150),
            'petal_length': np.random.normal(3.8, 1.8, 150),
            'petal_width': np.random.normal(1.2, 0.8, 150),
            'species': ['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50
        }
        df = pd.DataFrame(data)
        
        data_file = tmp_path / "test_data.csv"
        df.to_csv(data_file, index=False)
        return str(data_file)
    
    def test_pipeline_initialization(self, config):
        """Test pipeline initialization."""
        pipeline = IrisPipeline(config)
        assert pipeline.config == config
        assert isinstance(pipeline.trainer, IrisModelTrainer)
        assert isinstance(pipeline.deployer, VertexAIDeployer)
    
    def test_run_training_pipeline(self, config, sample_data, tmp_path):
        """Test the complete training pipeline."""
        config.data_path = sample_data
        config.artifacts_dir = str(tmp_path / "artifacts")
        
        pipeline = IrisPipeline(config)
        model_path = pipeline.run_training_pipeline()
        
        # Check if model file exists
        assert Path(model_path).exists()
        
        # Check if metrics file exists
        metrics_file = Path(config.artifacts_dir) / "metrics.json"
        assert metrics_file.exists()
        
        # Validate model can make predictions
        model = joblib.load(model_path)
        sample_input = [[5.1, 3.5, 1.4, 0.2]]
        prediction = model.predict(sample_input)
        assert len(prediction) == 1
        assert prediction[0] in ['setosa', 'versicolor', 'virginica']


class TestDataValidation:
    """Test cases for data validation and quality checks."""
    
    def test_data_schema_validation(self):
        """Test data schema validation."""
        # Valid data
        valid_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7],
            'sepal_width': [3.5, 3.0, 3.2],
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2],
            'species': ['setosa', 'setosa', 'setosa']
        })
        
        required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        assert all(col in valid_data.columns for col in required_columns)
        
        # Check data types
        numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(valid_data[col])
    
    def test_data_quality_checks(self):
        """Test data quality validation."""
        # Data with quality issues
        problematic_data = pd.DataFrame({
            'sepal_length': [5.1, np.nan, -1.0],  # Missing and negative values
            'sepal_width': [3.5, 3.0, 100.0],    # Outlier
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2],
            'species': ['setosa', 'unknown', 'setosa']  # Invalid species
        })
        
        # Check for missing values
        missing_values = problematic_data.isnull().sum()
        assert missing_values['sepal_length'] > 0
        
        # Check for negative values
        negative_values = (problematic_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] < 0).any()
        assert negative_values.any()
        
        # Check for valid species
        valid_species = {'setosa', 'versicolor', 'virginica'}
        invalid_species = set(problematic_data['species']) - valid_species
        assert len(invalid_species) > 0
    
    def test_data_distribution_validation(self):
        """Test data distribution validation."""
        np.random.seed(42)
        
        # Generate data with known properties
        data = pd.DataFrame({
            'sepal_length': np.random.normal(5.8, 0.8, 150),
            'sepal_width': np.random.normal(3.0, 0.4, 150),
            'petal_length': np.random.normal(3.8, 1.8, 150),
            'petal_width': np.random.normal(1.2, 0.8, 150),
            'species': ['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50
        })
        
        # Check species distribution
        species_counts = data['species'].value_counts()
        assert len(species_counts) == 3
        assert all(count == 50 for count in species_counts.values)
        
        # Check feature ranges (basic sanity checks)
        assert data['sepal_length'].min() > 0
        assert data['sepal_width'].min() > 0
        assert data['petal_length'].min() >= 0
        assert data['petal_width'].min() >= 0


class TestModelPerformance:
    """Test cases for model performance validation."""
    
    @pytest.fixture
    def trained_model_and_data(self):
        """Create a trained model with test data."""
        np.random.seed(42)
        
        # Generate synthetic but realistic iris data
        data = {
            'sepal_length': np.random.normal(5.8, 0.8, 150),
            'sepal_width': np.random.normal(3.0, 0.4, 150),
            'petal_length': np.random.normal(3.8, 1.8, 150),
            'petal_width': np.random.normal(1.2, 0.8, 150),
            'species': ['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50
        }
        df = pd.DataFrame(data)
        
        # Train model
        config = Config()
        trainer = IrisModelTrainer(config)
        X_train, y_train, X_test, y_test = trainer.prepare_data(df)
        trainer.train_model(X_train, y_train)
        metrics = trainer.evaluate_model(X_test, y_test)
        
        return trainer.model, X_test, y_test, metrics
    
    def test_model_accuracy_threshold(self, trained_model_and_data):
        """Test that model meets minimum accuracy threshold."""
        model, X_test, y_test, metrics = trained_model_and_data
        
        # Model should achieve at least 70% accuracy on iris dataset
        assert metrics['accuracy'] >= 0.7, f"Model accuracy {metrics['accuracy']:.3f} below threshold"
    
    def test_model_predictions_format(self, trained_model_and_data):
        """Test that model predictions are in the correct format."""
        model, X_test, y_test, metrics = trained_model_and_data
        
        predictions = model.predict(X_test)
        
        # Check prediction format
        assert len(predictions) == len(y_test)
        assert all(pred in ['setosa', 'versicolor', 'virginica'] for pred in predictions)
    
    def test_model_consistency(self, trained_model_and_data):
        """Test that model predictions are consistent."""
        model, X_test, y_test, metrics = trained_model_and_data
        
        # Same input should give same output
        sample_input = X_test.iloc[[0]]
        pred1 = model.predict(sample_input)[0]
        pred2 = model.predict(sample_input)[0]
        
        assert pred1 == pred2, "Model predictions are not consistent"
    
    def test_model_prediction_probabilities(self, trained_model_and_data):
        """Test model prediction probabilities."""
        model, X_test, y_test, metrics = trained_model_and_data
        
        # Get prediction probabilities
        probabilities = model.predict_proba(X_test)
        
        # Check shape and properties
        assert probabilities.shape == (len(X_test), 3)  # 3 classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probabilities >= 0)  # All probabilities non-negative


class TestIntegrationScenarios:
    """Integration tests for end-to-end scenarios."""
    
    def test_full_pipeline_integration(self, tmp_path):
        """Test complete pipeline from data to trained model."""
        # Create test data
        np.random.seed(42)
        data = {
            'sepal_length': np.random.normal(5.8, 0.8, 150),
            'sepal_width': np.random.normal(3.0, 0.4, 150),
            'petal_length': np.random.normal(3.8, 1.8, 150),
            'petal_width': np.random.normal(1.2, 0.8, 150),
            'species': ['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50
        }
        df = pd.DataFrame(data)
        
        # Setup temporary files
        data_file = tmp_path / "iris.csv"
        df.to_csv(data_file, index=False)
        
        artifacts_dir = tmp_path / "artifacts"
        
        # Configure pipeline
        config = Config(
            data_path=str(data_file),
            artifacts_dir=str(artifacts_dir),
            test_size=0.3,
            random_state=42
        )
        
        # Run training pipeline
        pipeline = IrisPipeline(config)
        model_path = pipeline.run_training_pipeline()
        
        # Verify outputs
        assert Path(model_path).exists()
        assert (artifacts_dir / "metrics.json").exists()
        
        # Load and test the saved model
        model = joblib.load(model_path)
        
        # Test prediction on new data
        new_sample = [[5.1, 3.5, 1.4, 0.2]]
        prediction = model.predict(new_sample)
        assert len(prediction) == 1
        assert prediction[0] in ['setosa', 'versicolor', 'virginica']
    
    def test_model_reproducibility(self, tmp_path):
        """Test that training is reproducible with same random seed."""
        # Create identical datasets
        np.random.seed(42)
        data = {
            'sepal_length': np.random.normal(5.8, 0.8, 100),
            'sepal_width': np.random.normal(3.0, 0.4, 100),
            'petal_length': np.random.normal(3.8, 1.8, 100),
            'petal_width': np.random.normal(1.2, 0.8, 100),
            'species': ['setosa'] * 33 + ['versicolor'] * 33 + ['virginica'] * 34
        }
        df = pd.DataFrame(data)
        
        data_file = tmp_path / "iris.csv"
        df.to_csv(data_file, index=False)
        
        # Train two models with same configuration
        config = Config(
            data_path=str(data_file),
            artifacts_dir=str(tmp_path / "artifacts1"),
            random_state=42
        )
        
        trainer1 = IrisModelTrainer(config)
        X_train, y_train, X_test, y_test = trainer1.prepare_data(df)
        trainer1.train_model(X_train, y_train)
        metrics1 = trainer1.evaluate_model(X_test, y_test)
        
        config.artifacts_dir = str(tmp_path / "artifacts2")
        trainer2 = IrisModelTrainer(config)
        trainer2.train_model(X_train, y_train)
        metrics2 = trainer2.evaluate_model(X_test, y_test)
        
        # Models should have identical performance
        assert metrics1['accuracy'] == metrics2['accuracy']
        
        # Predictions should be identical
        test_sample = X_test.iloc[[0]]
        pred1 = trainer1.model.predict(test_sample)[0]
        pred2 = trainer2.model.predict(test_sample)[0]
        assert pred1 == pred2


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"]) 
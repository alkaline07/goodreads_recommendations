"""
Test cases for register_bqml_models.py
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.register_bqml_models import RegisterBQMLModels


class TestRegisterBQMLModels:
    """Test cases for RegisterBQMLModels class"""
    
    def test_init(self):
        """Test initialization"""
        registrar = RegisterBQMLModels()
        
        assert "BOOSTED_TREE" in registrar.MODEL_REGISTRY_NAMES
        assert "BOOSTED_TREE" in registrar.MODEL_PREFIXES
    
    def test_get_serving_image(self):
        """Test getting serving image"""
        registrar = RegisterBQMLModels()
        
        image = registrar.get_serving_image("BOOSTED_TREE")
        
        assert "bqml_xgboost" in image
        assert isinstance(image, str)
    
    def test_get_serving_image_unknown(self):
        """Test getting serving image for unknown type"""
        registrar = RegisterBQMLModels()
        
        with pytest.raises(ValueError):
            registrar.get_serving_image("UNKNOWN_TYPE")
    
    @patch('src.register_bqml_models.bigquery.Client')
    def test_find_latest_model(self, mock_client_class):
        """Test finding latest model"""
        mock_client = Mock()
        mock_model = Mock()
        mock_model.model_id = "boosted_tree_regressor_model_20250101"
        mock_model.created = datetime(2025, 1, 1)
        mock_client.list_models.return_value = [mock_model]
        mock_client_class.return_value = mock_client
        
        registrar = RegisterBQMLModels()
        
        result = registrar.find_latest_model(
            mock_client,
            "test-project",
            "books",
            "boosted_tree_regressor_model"
        )
        
        assert result is not None
        assert "boosted_tree_regressor_model" in result
    
    @patch('src.register_bqml_models.bigquery.Client')
    def test_find_latest_model_not_found(self, mock_client_class):
        """Test finding latest model when none exists"""
        mock_client = Mock()
        mock_client.list_models.return_value = []
        mock_client_class.return_value = mock_client
        
        registrar = RegisterBQMLModels()
        
        result = registrar.find_latest_model(
            mock_client,
            "test-project",
            "books",
            "nonexistent_model"
        )
        
        assert result is None
    
    @patch('src.register_bqml_models.aiplatform')
    def test_register_model_in_vertex_ai(self, mock_aiplatform):
        """Test registering model in Vertex AI"""
        mock_model = Mock()
        mock_model.resource_name = "projects/test/locations/us-central1/models/123"
        mock_model.versioned_resource_name = "projects/test/locations/us-central1/models/123@1"
        mock_aiplatform.Model.upload.return_value = mock_model
        
        existing_models = []
        mock_aiplatform.Model.list.return_value = existing_models
        
        registrar = RegisterBQMLModels()
        
        registrar.register_model_in_vertex_ai(
            "test-project",
            "us-central1",
            "test-project.books.test_model",
            "test-model",
            "BOOSTED_TREE"
        )
        
        mock_aiplatform.Model.upload.assert_called_once()
    
    @patch('src.register_bqml_models.bigquery.Client')
    @patch('src.register_bqml_models.aiplatform')
    def test_main(self, mock_aiplatform, mock_client_class):
        """Test main method"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_model = Mock()
        mock_model.model_id = "boosted_tree_regressor_model_20250101"
        mock_model.created = datetime(2025, 1, 1)
        mock_client.list_models.return_value = [mock_model]
        mock_client_class.return_value = mock_client
        
        mock_aiplatform_model = Mock()
        mock_aiplatform_model.resource_name = "test-resource"
        mock_aiplatform.Model.upload.return_value = mock_aiplatform_model
        mock_aiplatform.Model.list.return_value = []
        
        registrar = RegisterBQMLModels()
        
        with patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'}):
            registrar.main()
            # Should not raise exception
            assert True


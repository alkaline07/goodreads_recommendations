"""
Comprehensive test cases for register_bqml_models.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.register_bqml_models import RegisterBQMLModels


class TestRegisterBQMLModels:
    """Comprehensive test cases for RegisterBQMLModels class"""
    
    def test_init(self):
        """Test initialization"""
        registrar = RegisterBQMLModels()
        
        assert "BOOSTED_TREE" in registrar.MODEL_REGISTRY_NAMES
        assert "BOOSTED_TREE" in registrar.MODEL_PREFIXES
        assert isinstance(registrar.MODEL_REGISTRY_NAMES, dict)
        assert isinstance(registrar.MODEL_PREFIXES, dict)
    
    def test_get_serving_image_boosted_tree(self):
        """Test getting serving image for boosted tree"""
        registrar = RegisterBQMLModels()
        
        image = registrar.get_serving_image("BOOSTED_TREE")
        
        assert "bqml_xgboost" in image
        assert isinstance(image, str)
        assert len(image) > 0
    
    def test_get_serving_image_unknown(self):
        """Test getting serving image for unknown type"""
        registrar = RegisterBQMLModels()
        
        with pytest.raises(ValueError):
            registrar.get_serving_image("UNKNOWN_TYPE")
    
    def test_get_serving_image_empty_string(self):
        """Test getting serving image with empty string"""
        registrar = RegisterBQMLModels()
        
        with pytest.raises(ValueError):
            registrar.get_serving_image("")
    
    @patch('src.register_bqml_models.bigquery.Client')
    def test_find_latest_model_success(self, mock_client_class):
        """Test finding latest model successfully"""
        mock_client = Mock()
        mock_model1 = Mock()
        mock_model1.model_id = "boosted_tree_regressor_model_20250101"
        mock_model1.created = datetime(2025, 1, 1)
        mock_model2 = Mock()
        mock_model2.model_id = "boosted_tree_regressor_model_20250102"
        mock_model2.created = datetime(2025, 1, 2)
        mock_client.list_models.return_value = [mock_model1, mock_model2]
        mock_client_class.return_value = mock_client
        
        registrar = RegisterBQMLModels()
        
        result = registrar.find_latest_model(
            mock_client,
            "test-project",
            "books",
            "boosted_tree_regressor_model"
        )
        
        assert result is not None
        assert "boosted_tree_regressor_model_20250102" in result
        assert "test-project" in result
        assert "books" in result
    
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
    
    @patch('src.register_bqml_models.bigquery.Client')
    def test_find_latest_model_no_matching_prefix(self, mock_client_class):
        """Test finding latest model when no models match prefix"""
        mock_client = Mock()
        mock_model = Mock()
        mock_model.model_id = "other_model_20250101"
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
        
        assert result is None
    
    @patch('src.register_bqml_models.bigquery.Client')
    def test_find_latest_model_exception(self, mock_client_class):
        """Test exception handling in find_latest_model"""
        mock_client = Mock()
        mock_client.list_models.side_effect = Exception("List failed")
        mock_client_class.return_value = mock_client
        
        registrar = RegisterBQMLModels()
        
        result = registrar.find_latest_model(
            mock_client,
            "test-project",
            "books",
            "test_model"
        )
        
        assert result is None
    
    @patch('src.register_bqml_models.bigquery.Client')
    def test_find_latest_model_empty_dataset(self, mock_client_class):
        """Test finding latest model in empty dataset"""
        mock_client = Mock()
        mock_client.list_models.return_value = []
        mock_client_class.return_value = mock_client
        
        registrar = RegisterBQMLModels()
        
        result = registrar.find_latest_model(
            mock_client,
            "test-project",
            "books",
            "test_model"
        )
        
        assert result is None
    
    @patch('src.register_bqml_models.aiplatform')
    def test_register_model_in_vertex_ai_new_model(self, mock_aiplatform):
        """Test registering new model in Vertex AI"""
        mock_model = Mock()
        mock_model.resource_name = "projects/test/locations/us-central1/models/123"
        mock_model.versioned_resource_name = "projects/test/locations/us-central1/models/123@1"
        mock_aiplatform.Model.upload.return_value = mock_model
        
        existing_models = []
        mock_aiplatform.Model.list.return_value = existing_models
        
        registrar = RegisterBQMLModels()
        
        result = registrar.register_model_in_vertex_ai(
            "test-project",
            "us-central1",
            "test-project.books.test_model",
            "test-model",
            "BOOSTED_TREE"
        )
        
        mock_aiplatform.Model.upload.assert_called_once()
        assert result is not None
    
    @patch('src.register_bqml_models.aiplatform')
    def test_register_model_in_vertex_ai_existing_model(self, mock_aiplatform):
        """Test registering model as new version of existing model"""
        mock_model = Mock()
        mock_model.resource_name = "projects/test/locations/us-central1/models/123"
        mock_model.versioned_resource_name = "projects/test/locations/us-central1/models/123@1"
        mock_aiplatform.Model.upload.return_value = mock_model
        
        existing_model = Mock()
        existing_model.resource_name = "projects/test/locations/us-central1/models/123"
        mock_aiplatform.Model.list.return_value = [existing_model]
        
        registrar = RegisterBQMLModels()
        
        result = registrar.register_model_in_vertex_ai(
            "test-project",
            "us-central1",
            "test-project.books.test_model",
            "test-model",
            "BOOSTED_TREE"
        )
        
        mock_aiplatform.Model.upload.assert_called_once()
        assert result is not None
    
    @patch('src.register_bqml_models.aiplatform')
    def test_register_model_in_vertex_ai_with_set_as_default(self, mock_aiplatform):
        """Test registering model and setting as default"""
        mock_model = Mock()
        mock_model.resource_name = "projects/test/locations/us-central1/models/123"
        mock_aiplatform.Model.upload.return_value = mock_model
        mock_aiplatform.Model.list.return_value = []
        
        registrar = RegisterBQMLModels()
        
        result = registrar.register_model_in_vertex_ai(
            "test-project",
            "us-central1",
            "test-project.books.test_model",
            "test-model",
            "BOOSTED_TREE",
            set_as_default=True
        )
        
        # Check that is_default_version=True was passed
        call_kwargs = mock_aiplatform.Model.upload.call_args[1]
        assert call_kwargs.get('is_default_version') is True
    
    @patch('src.register_bqml_models.aiplatform')
    def test_register_model_in_vertex_ai_exception(self, mock_aiplatform):
        """Test exception handling in register_model_in_vertex_ai"""
        mock_aiplatform.Model.upload.side_effect = Exception("Upload failed")
        mock_aiplatform.Model.list.return_value = []
        
        registrar = RegisterBQMLModels()
        
        result = registrar.register_model_in_vertex_ai(
            "test-project",
            "us-central1",
            "test-project.books.test_model",
            "test-model",
            "BOOSTED_TREE"
        )
        
        assert result is None
    
    @patch('src.register_bqml_models.aiplatform')
    def test_register_model_in_vertex_ai_list_exception(self, mock_aiplatform):
        """Test exception handling when listing existing models"""
        mock_model = Mock()
        mock_model.resource_name = "projects/test/locations/us-central1/models/123"
        mock_aiplatform.Model.upload.return_value = mock_model
        mock_aiplatform.Model.list.side_effect = Exception("List failed")
        
        registrar = RegisterBQMLModels()
        
        # Should still register as new model
        result = registrar.register_model_in_vertex_ai(
            "test-project",
            "us-central1",
            "test-project.books.test_model",
            "test-model",
            "BOOSTED_TREE"
        )
        
        assert result is not None
    
    @patch('src.register_bqml_models.bigquery.Client')
    @patch('src.register_bqml_models.aiplatform')
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_main_success(self, mock_aiplatform, mock_client_class):
        """Test main method successfully"""
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
        
        registrar.main()
        
        # Should not raise exception
        assert True
    
    @patch('src.register_bqml_models.bigquery.Client')
    @patch('src.register_bqml_models.aiplatform')
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_main_no_models_found(self, mock_aiplatform, mock_client_class):
        """Test main method when no models found"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.list_models.return_value = []
        mock_client_class.return_value = mock_client
        
        registrar = RegisterBQMLModels()
        
        registrar.main()
        
        # Should not raise exception, just skip registration
        assert True
    
    @patch('src.register_bqml_models.bigquery.Client')
    @patch('src.register_bqml_models.aiplatform')
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_main_registration_exception(self, mock_aiplatform, mock_client_class):
        """Test main method when registration fails"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_model = Mock()
        mock_model.model_id = "boosted_tree_regressor_model_20250101"
        mock_model.created = datetime(2025, 1, 1)
        mock_client.list_models.return_value = [mock_model]
        mock_client_class.return_value = mock_client
        
        mock_aiplatform.Model.upload.side_effect = Exception("Upload failed")
        mock_aiplatform.Model.list.return_value = []
        
        registrar = RegisterBQMLModels()
        
        registrar.main()
        
        # Should not raise exception, just continue
        assert True
    
    @patch('src.register_bqml_models.bigquery.Client')
    @patch('src.register_bqml_models.aiplatform')
    def test_main_without_airflow_home(self, mock_aiplatform, mock_client_class):
        """Test main method without AIRFLOW_HOME"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.list_models.return_value = []
        mock_client_class.return_value = mock_client
        
        registrar = RegisterBQMLModels()
        
        with patch.dict('os.environ', {}, clear=True):
            registrar.main()
            # Should not raise exception
            assert True

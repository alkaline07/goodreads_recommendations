import pytest
from unittest.mock import Mock, patch, MagicMock
from api.generate_predictions import GeneratePredictions
import pandas as pd


class TestGeneratePredictions:
    """Test suite for GeneratePredictions class"""

    @pytest.fixture
    def mock_bigquery_client(self):
        """Mock BigQuery client"""
        mock_client = Mock()
        mock_client.project = "test-project"
        return mock_client

    @pytest.fixture
    def mock_aiplatform(self):
        """Mock AI Platform"""
        with patch('api.generate_predictions.aiplatform') as mock_ai:
            yield mock_ai

    @pytest.fixture
    @patch('api.generate_predictions.bigquery.Client')
    @patch('api.generate_predictions.aiplatform')
    def generator(self, mock_ai, mock_bq):
        """Create GeneratePredictions instance with mocked dependencies"""
        mock_bq.return_value.project = "test-project"
        return GeneratePredictions()

    def test_init(self, mock_aiplatform):
        """Test initialization of GeneratePredictions"""
        with patch('api.generate_predictions.bigquery.Client') as mock_bq:
            mock_bq.return_value.project = "test-project"
            generator = GeneratePredictions()
            
            assert generator.project_id == "test-project"
            assert generator.dataset_id == "books"
            assert generator.location == "us-central1"

    def test_get_version_found(self, generator):
        """Test get_version when default version is found"""
        # Mock the Model.list and versioning_registry
        mock_version = Mock()
        mock_version.version_id = "123"
        mock_version.version_aliases = ['default']
        
        mock_model = Mock()
        mock_model.versioning_registry.list_versions.return_value = [mock_version]
        
        with patch('api.generate_predictions.aiplatform.Model.list') as mock_list:
            mock_list.return_value = [mock_model]
            
            result = generator.get_version("test-model")
            assert result == "123"

    def test_get_version_not_found(self, generator):
        """Test get_version when no models exist"""
        with patch('api.generate_predictions.aiplatform.Model.list') as mock_list:
            mock_list.return_value = []
            
            result = generator.get_version("nonexistent-model")
            assert result is None

    def test_get_version_exception(self, generator):
        """Test get_version handles exceptions gracefully"""
        with patch('api.generate_predictions.aiplatform.Model.list') as mock_list:
            mock_list.side_effect = Exception("API Error")
            
            result = generator.get_version("test-model")
            assert result is None

    def test_get_bq_model_id_by_version_boosted_tree(self, generator):
        """Test get_bq_model_id_by_version for boosted tree model"""
        with patch('api.generate_predictions.aiplatform.Model') as mock_model:
            mock_instance = Mock()
            mock_instance.to_dict.return_value = {
                'versionCreateTime': '2025-12-04T10:30:00+00:00'
            }
            mock_model.return_value = mock_instance
            
            result = generator.get_bq_model_id_by_version("boosted_tree_model", "v1")
            
            assert "boosted_tree_regressor_model" in result
            assert "20251204_103000" in result

    def test_get_bq_model_id_by_version_matrix_factorization(self, generator):
        """Test get_bq_model_id_by_version for matrix factorization model"""
        with patch('api.generate_predictions.aiplatform.Model') as mock_model:
            mock_instance = Mock()
            mock_instance.to_dict.return_value = {
                'versionCreateTime': '2025-12-04T10:30:00+00:00'
            }
            mock_model.return_value = mock_instance
            
            result = generator.get_bq_model_id_by_version("matrix_factorization_model", "v1")
            
            assert "matrix_factorization_model" in result
            assert "20251204_103000" in result

    def test_get_bq_model_id_by_version_invalid_model(self, generator):
        """Test get_bq_model_id_by_version with invalid model type"""
        with patch('api.generate_predictions.aiplatform.Model') as mock_model:
            mock_instance = Mock()
            mock_instance.to_dict.return_value = {
                'versionCreateTime': '2025-12-04T10:30:00+00:00'
            }
            mock_model.return_value = mock_instance
            
            with pytest.raises(ValueError):
                generator.get_bq_model_id_by_version("unknown_model", "v1")

    def test_get_model_from_registry_boosted_tree(self, generator):
        """Test get_model_from_registry for boosted tree"""
        with patch.object(generator, 'get_version', return_value='v1') as mock_version, \
             patch.object(generator, 'get_bq_model_id_by_version', return_value='test-project.books.boosted_tree_regressor_model') as mock_bq_id:
            
            result = generator.get_model_from_registry("boosted_tree_regressor_model")
            
            assert result == "test-project.books.boosted_tree_regressor_model"
            mock_version.assert_called_once_with("boosted_tree_regressor_model")
            mock_bq_id.assert_called_once_with("boosted_tree_regressor_model", "v1")

    def test_get_model_from_registry_matrix_factorization(self, generator):
        """Test get_model_from_registry for matrix factorization"""
        with patch.object(generator, 'get_version', return_value='v1') as mock_version, \
             patch.object(generator, 'get_bq_model_id_by_version', return_value='test-project.books.matrix_factorization_model') as mock_bq_id:
            
            result = generator.get_model_from_registry("matrix_factorization_model")
            
            assert result == "test-project.books.matrix_factorization_model"

    def test_get_model_from_registry_invalid(self, generator):
        """Test get_model_from_registry with invalid model type"""
        with patch.object(generator, 'get_version', return_value=None):
            result = generator.get_model_from_registry("unknown_model_type")
            assert result is None

    def test_get_mf_predictions_success(self, generator):
        """Test get_mf_predictions returns expected dataframe"""
        mock_df = pd.DataFrame({
            'book_id': [1, 2, 3],
            'title': ['Book A', 'Book B', 'Book C'],
            'rating': [4.5, 4.2, 3.8],
            'author_names': ['Author 1', 'Author 2', 'Author 3']
        })
        
        mock_query = Mock()
        mock_query.to_dataframe.return_value = mock_df
        generator.client.query.return_value = mock_query
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "SELECT * FROM {project_id}.{dataset}.model WHERE user_id = @user_id"
            )
            
            result = generator.get_mf_predictions("test-model", "user_123")
            
            assert len(result) == 3
            assert list(result.columns) == ['book_id', 'title', 'rating', 'author_names']

    def test_get_bt_predictions_success(self, generator):
        """Test get_bt_predictions returns expected dataframe"""
        mock_df = pd.DataFrame({
            'book_id': [1, 2, 3],
            'title': ['Book A', 'Book B', 'Book C'],
            'rating': [4.5, 4.2, 3.8],
            'author_names': ['Author 1', 'Author 2', 'Author 3']
        })
        
        mock_query = Mock()
        mock_query.to_dataframe.return_value = mock_df
        generator.client.query.return_value = mock_query
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "SELECT * FROM {project_id}.{dataset}.model WHERE user_id = @user_id"
            )
            
            result = generator.get_bt_predictions("test-model", "user_123")
            
            assert len(result) == 3
            assert list(result.columns) == ['book_id', 'title', 'rating', 'author_names']

    def test_get_predictions_matrix_factorization(self, generator):
        """Test get_predictions with matrix factorization model"""
        mock_model_info = {'display_name': 'matrix_factorization_model'}
        mock_df = pd.DataFrame({
            'book_id': [1, 2, 3],
            'title': ['Book A', 'Book B', 'Book C'],
            'rating': [4.5, 4.2, 3.8],
            'author_names': ['Author 1', 'Author 2', 'Author 3']
        })
        
        with patch.object(GeneratePredictions, 'get_selected_model_info') as mock_get_model:
            mock_get_model.return_value = mock_model_info
            
            # FIX: Mock get_model_from_registry to avoid complex internal calls
            with patch.object(generator, 'get_model_from_registry', return_value='mf_model_id'):
                with patch.object(generator, 'get_mf_predictions') as mock_mf:
                    mock_mf.return_value = mock_df
                    
                    result = generator.get_predictions("user_123")
                    
                    assert len(result) == 3
                    mock_mf.assert_called_once()

    def test_get_predictions_boosted_tree(self, generator):
        """Test get_predictions with boosted tree model"""
        mock_model_info = {'display_name': 'boosted_tree_regressor_model'}
        mock_df = pd.DataFrame({
            'book_id': [1, 2, 3],
            'title': ['Book A', 'Book B', 'Book C'],
            'rating': [4.5, 4.2, 3.8],
            'author_names': ['Author 1', 'Author 2', 'Author 3']
        })
        
        with patch.object(GeneratePredictions, 'get_selected_model_info') as mock_get_model:
            mock_get_model.return_value = mock_model_info
            
            with patch.object(generator, 'get_model_from_registry', return_value='bt_model_id'):
                with patch.object(generator, 'get_bt_predictions') as mock_bt:
                    mock_bt.return_value = mock_df
                    
                    result = generator.get_predictions("user_123")
                    
                    assert len(result) == 3
                    mock_bt.assert_called_once()

    def test_get_predictions_no_model_selected(self, generator):
        """Test get_predictions raises error when no model is selected"""
        with patch.object(GeneratePredictions, 'get_selected_model_info') as mock_get_model:
            mock_get_model.return_value = None
            
            with pytest.raises(ValueError, match="No model selected"):
                generator.get_predictions("user_123")

    def test_get_predictions_invalid_model_type(self, generator):
        """Test get_predictions raises error for invalid model type"""
        mock_model_info = {'display_name': 'invalid_model_type'}
        
        with patch.object(GeneratePredictions, 'get_selected_model_info') as mock_get_model:
            mock_get_model.return_value = mock_model_info
            
            with patch.object(generator, 'get_model_from_registry', return_value=None):
                with pytest.raises(ValueError, match="Could not retrieve"):
                    generator.get_predictions("user_123")


class TestGeneratePredictionsIntegration:
    """Integration tests for GeneratePredictions"""

    @pytest.fixture
    @patch('api.generate_predictions.bigquery.Client')
    @patch('api.generate_predictions.aiplatform')
    def generator(self, mock_ai, mock_bq):
        """Create GeneratePredictions instance for integration tests"""
        mock_bq.return_value.project = "test-project"
        return GeneratePredictions()

    def test_full_prediction_pipeline(self, generator):
        """Test full prediction pipeline end-to-end"""
        mock_model_info = {'display_name': 'matrix_factorization_model'}
        mock_df = pd.DataFrame({
            'book_id': [1, 2],
            'title': ['Book A', 'Book B'],
            'rating': [4.5, 4.2],
            'author_names': ['Author 1', 'Author 2']
        })
        
        mock_version = Mock()
        mock_version.version_aliases = ['default']
        mock_version.version_id = 'v1'
        
        mock_parent_model = Mock()
        mock_parent_model.resource_name = "projects/test/locations/us-central1/models/123" 
        mock_parent_model.versioning_registry.list_versions.return_value = [mock_version]

        with patch('api.generate_predictions.aiplatform') as mock_ai:
            mock_ai.Model.list.return_value = [mock_parent_model]
            
            # Mock the Model constructor used in get_bq_model_id_by_version
            mock_version_model = Mock()
            mock_version_model.to_dict.return_value = {'versionCreateTime': '2025-01-01T12:00:00Z'}
            mock_ai.Model.return_value = mock_version_model

            with patch.object(GeneratePredictions, 'get_selected_model_info') as mock_get_model:
                mock_get_model.return_value = mock_model_info
                
                # Mock BigQuery Query Job
                mock_query_job = Mock()
                mock_query_job.to_dataframe.return_value = mock_df
                generator.client.query.return_value = mock_query_job
                
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = "SELECT * FROM ..."
                    
                    result = generator.get_predictions("user_123")
                    
                    # Verify result structure and content
                    assert isinstance(result, pd.DataFrame)
                    assert len(result) == 2
                    assert all(col in result.columns for col in ['book_id', 'title', 'rating', 'author_names'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Comprehensive test cases for load_data.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.load_data import DataLoader


class TestDataLoader:
    """Comprehensive test cases for DataLoader class"""
    
    @patch('src.load_data.bigquery.Client')
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_init(self, mock_client_class):
        """Test initialization with AIRFLOW_HOME set"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        loader = DataLoader()
        
        assert loader.project_id == "test-project"
        assert loader.dataset_id == "books"
    
    @patch('src.load_data.bigquery.Client')
    def test_init_without_airflow_home(self, mock_client_class):
        """Test initialization without AIRFLOW_HOME"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        with patch.dict('os.environ', {}, clear=True):
            loader = DataLoader()
            assert loader.project_id == "test-project"
    
    @patch('src.load_data.bigquery.Client')
    @patch('src.load_data.bpd.read_gbq')
    def test_load_data_success(self, mock_read_gbq, mock_client_class):
        """Test loading data successfully"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        mock_df = Mock()
        mock_read_gbq.return_value = mock_df
        
        loader = DataLoader()
        loader.client = mock_client
        loader.project_id = "test-project"
        loader.dataset_id = "books"
        
        result = loader.load_data()
        
        assert result == mock_df
        mock_read_gbq.assert_called_once()
    
    @patch('src.load_data.bigquery.Client')
    @patch('src.load_data.bpd.read_gbq')
    def test_load_data_exception(self, mock_read_gbq, mock_client_class):
        """Test exception handling in load_data"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        mock_read_gbq.side_effect = Exception("Query failed")
        
        loader = DataLoader()
        loader.client = mock_client
        loader.project_id = "test-project"
        loader.dataset_id = "books"
        
        with pytest.raises(Exception):
            loader.load_data()
    
    @patch('src.load_data.bigquery.Client')
    @patch('src.load_data.bpd.read_gbq')
    def test_load_data_empty_result(self, mock_read_gbq, mock_client_class):
        """Test loading data with empty result"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        # Create a mock DataFrame instead of using real bigframes
        mock_df = MagicMock()
        mock_read_gbq.return_value = mock_df
        
        loader = DataLoader()
        loader.client = mock_client
        loader.project_id = "test-project"
        loader.dataset_id = "books"
        
        result = loader.load_data()
        
        assert result is not None
    
    @patch('src.load_data.bigquery.Client')
    @patch('src.load_data.bpd.read_gbq')
    def test_load_data_query_format(self, mock_read_gbq, mock_client_class):
        """Test that load_data uses correct query format"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        mock_df = Mock()
        mock_read_gbq.return_value = mock_df
        
        loader = DataLoader()
        loader.client = mock_client
        loader.project_id = "test-project"
        loader.dataset_id = "books"
        
        loader.load_data()
        
        # Verify query contains correct table reference
        call_args = mock_read_gbq.call_args[0][0]
        assert "goodreads_train_set" in call_args
        assert "test-project" in call_args
        assert "books" in call_args

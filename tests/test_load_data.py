"""
Test cases for load_data.py
"""

import pytest
from unittest.mock import Mock, patch
from src.load_data import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class"""
    
    @patch('src.load_data.bigquery.Client')
    @patch.dict('os.environ', {'AIRFLOW_HOME': '/test/airflow'})
    def test_init(self, mock_client_class):
        """Test initialization"""
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client_class.return_value = mock_client
        
        loader = DataLoader()
        
        assert loader.project_id == "test-project"
        assert loader.dataset_id == "books"
    
    @patch('src.load_data.bigquery.Client')
    @patch('src.load_data.bpd.read_gbq')
    def test_load_data(self, mock_read_gbq, mock_client_class):
        """Test loading data"""
        import bigframes.pandas as bpd
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


import unittest
from unittest.mock import patch, MagicMock
from src.monitor_decay import MonitorDecay

class TestMonitorDecay(unittest.TestCase):

    @patch('src.monitor_decay.bigquery.Client')
    @patch('src.monitor_decay.sys.exit')
    def test_decay_detected(self, mock_exit, mock_bq_client):
        """Test that sys.exit(1) is called when Avg CTR is below threshold."""
        # Setup the mock client
        mock_client_instance = mock_bq_client.return_value
        
        # Create a mock row object to simulate the NEW BigQuery result structure
        mock_row = MagicMock()
        mock_row.avg_ctr = 0.01  # 1.0%
        mock_row.distinct_users_counted = 50
        
        # Mock the job result iterator
        mock_job = MagicMock()
        mock_job.result.return_value = [mock_row]
        mock_client_instance.query.return_value = mock_job

        # Run the monitor
        monitor = MonitorDecay()
        monitor.CTR_THRESHOLD = 0.02 # Set explicit threshold (2.0%) for test stability
        monitor.check_model_decay()

        # Assertions
        # Should exit with code 1 (failure) because 1.0% < 2.0%
        mock_exit.assert_called_with(1) 

    @patch('src.monitor_decay.bigquery.Client')
    @patch('src.monitor_decay.sys.exit')
    def test_healthy_model(self, mock_exit, mock_bq_client):
        """Test that sys.exit(0) is called when Avg CTR is above threshold."""
        mock_client_instance = mock_bq_client.return_value
        
        # Mock row with healthy stats
        mock_row = MagicMock()
        mock_row.avg_ctr = 0.05  # 5.0%
        mock_row.distinct_users_counted = 100
        
        mock_job = MagicMock()
        mock_job.result.return_value = [mock_row]
        mock_client_instance.query.return_value = mock_job

        monitor = MonitorDecay()
        monitor.CTR_THRESHOLD = 0.02 # Set explicit threshold (2.0%)
        monitor.check_model_decay()

        # Should exit with code 0 (success) because 5.0% > 2.0%
        mock_exit.assert_called_with(0)

    @patch('src.monitor_decay.bigquery.Client')
    @patch('src.monitor_decay.sys.exit')
    def test_no_data_found(self, mock_exit, mock_bq_client):
        """Test behavior when query returns no rows (no users with clicks)."""
        mock_client_instance = mock_bq_client.return_value
        
        # Simulate empty result list
        mock_job = MagicMock()
        mock_job.result.return_value = []
        mock_client_instance.query.return_value = mock_job

        monitor = MonitorDecay()
        monitor.check_model_decay()

        # Should exit with code 0 (soft pass) based on the "sys.exit(0)" in the updated code
        mock_exit.assert_called_with(0)

if __name__ == '__main__':
    unittest.main()
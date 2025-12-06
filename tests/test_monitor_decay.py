import unittest
from unittest.mock import patch, MagicMock
from src.monitor_decay import MonitorDecay

class TestMonitorDecay(unittest.TestCase):

    @patch('src.monitor_decay.bigquery.Client')
    @patch('src.monitor_decay.sys.exit')
    def test_decay_detected(self, mock_exit, mock_bq_client):
        """Test that sys.exit(1) is called when CTR is below threshold."""
        # Setup the mock client
        mock_client_instance = mock_bq_client.return_value
        
        # Create a mock row object to simulate BigQuery result
        mock_row = MagicMock()
        mock_row.views = 1000
        mock_row.clicks = 10
        mock_row.ctr = 0.01  # 1.0% which is < 1.5% threshold
        
        # Mock the job result iterator
        mock_job = MagicMock()
        mock_job.result.return_value = [mock_row]
        mock_client_instance.query.return_value = mock_job

        # Run the monitor
        monitor = MonitorDecay()
        monitor.check_model_decay()

        # Assertions
        # Should exit with code 1 (failure) due to decay
        mock_exit.assert_called_with(1) 

    @patch('src.monitor_decay.bigquery.Client')
    @patch('src.monitor_decay.sys.exit')
    def test_healthy_model(self, mock_exit, mock_bq_client):
        """Test that sys.exit(0) is called when CTR is above threshold."""
        mock_client_instance = mock_bq_client.return_value
        
        mock_row = MagicMock()
        mock_row.views = 1000
        mock_row.clicks = 50
        mock_row.ctr = 0.05  # 5.0% which is > 1.5% threshold
        
        mock_job = MagicMock()
        mock_job.result.return_value = [mock_row]
        mock_client_instance.query.return_value = mock_job

        monitor = MonitorDecay()
        monitor.check_model_decay()

        # Should exit with code 0 (success)
        mock_exit.assert_called_with(0)

    @patch('src.monitor_decay.bigquery.Client')
    @patch('src.monitor_decay.sys.exit')
    def test_no_data_found(self, mock_exit, mock_bq_client):
        """Test behavior when query returns no rows."""
        mock_client_instance = mock_bq_client.return_value
        
        # Simulate empty result list
        mock_job = MagicMock()
        mock_job.result.return_value = []
        mock_client_instance.query.return_value = mock_job

        monitor = MonitorDecay()
        monitor.check_model_decay()

        # Should NOT call sys.exit if no data is found, just return
        mock_exit.assert_not_called()

if __name__ == '__main__':
    unittest.main()
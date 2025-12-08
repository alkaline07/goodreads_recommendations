import unittest
from unittest.mock import patch, MagicMock
from google.api_core.exceptions import NotFound
from api.log_click_event import LogClickEvent

class TestLogClickEvent(unittest.TestCase):

    @patch('api.log_click_event.bigquery.Client')
    def test_create_table_if_not_exists(self, mock_bq_client):
        """Test that create_table is called if get_table raises NotFound."""
        mock_client_instance = mock_bq_client.return_value
        mock_client_instance.project = "test_project"
        
        # Simulate NotFound exception when getting table
        mock_client_instance.get_table.side_effect = NotFound("Table not found")

        logger = LogClickEvent()
        # We manually call the internal method to test it specifically
        logger._create_table_if_not_exists()

        # Assert create_table was called
        mock_client_instance.create_table.assert_called_once()

    @patch('api.log_click_event.bigquery.Client')
    def test_log_valid_event(self, mock_bq_client):
        """Test logging a valid event inserts rows."""
        mock_client_instance = mock_bq_client.return_value
        # Simulate successful insertion (returns empty list of errors)
        mock_client_instance.insert_rows_json.return_value = []

        logger = LogClickEvent()
        result = logger.log_user_event("user_123", "book_456", "click")

        self.assertTrue(result)
        mock_client_instance.insert_rows_json.assert_called_once()

    @patch('api.log_click_event.bigquery.Client')
    def test_invalid_event_type(self, mock_bq_client):
        """Test that ValueError is raised for invalid event types."""
        logger = LogClickEvent()
        
        # "purchase" is not in the allowed list
        with self.assertRaises(ValueError):
            logger.log_user_event("user_123", "book_456", "purchase")

if __name__ == '__main__':
    unittest.main()
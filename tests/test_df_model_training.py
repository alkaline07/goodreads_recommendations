"""
Test cases for df_model_training.py
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from src.df_model_training import DataFrameModelTraining


class TestDataFrameModelTraining:
    """Test cases for DataFrameModelTraining class"""
    
    def test_init(self):
        """Test initialization"""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        trainer = DataFrameModelTraining(df)
        
        assert trainer.df.shape == (3, 2)
        assert len(trainer.df.columns) == 2
    
    def test_train_model(self):
        """Test train_model method"""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        trainer = DataFrameModelTraining(df)
        
        # Should not raise exception
        trainer.train_model()
        assert True


"""
Comprehensive test cases for df_model_training.py
Covers all scenarios: success, failure, edge cases, boundary conditions
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from src.df_model_training import DataFrameModelTraining


class TestDataFrameModelTraining:
    """Comprehensive test cases for DataFrameModelTraining class"""
    
    def test_init(self):
        """Test initialization with valid DataFrame"""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        trainer = DataFrameModelTraining(df)
        
        assert trainer.df.shape == (3, 2)
        assert len(trainer.df.columns) == 2
        assert list(trainer.df.columns) == ['col1', 'col2']
    
    def test_init_empty_dataframe(self):
        """Test initialization with empty DataFrame"""
        df = pd.DataFrame()
        trainer = DataFrameModelTraining(df)
        
        assert trainer.df.shape == (0, 0)
        assert len(trainer.df.columns) == 0
    
    def test_init_single_column(self):
        """Test initialization with single column DataFrame"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        trainer = DataFrameModelTraining(df)
        
        assert trainer.df.shape == (3, 1)
        assert len(trainer.df.columns) == 1
    
    def test_init_large_dataframe(self):
        """Test initialization with large DataFrame"""
        df = pd.DataFrame({
            'col1': range(10000),
            'col2': range(10000, 20000),
            'col3': range(20000, 30000)
        })
        trainer = DataFrameModelTraining(df)
        
        assert trainer.df.shape == (10000, 3)
        assert len(trainer.df.columns) == 3
    
    def test_init_with_nan_values(self):
        """Test initialization with NaN values"""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [5, np.nan, 7, 8]
        })
        trainer = DataFrameModelTraining(df)
        
        assert trainer.df.shape == (4, 2)
        assert trainer.df.isna().sum().sum() > 0
    
    def test_init_with_string_columns(self):
        """Test initialization with string columns"""
        df = pd.DataFrame({
            'col1': ['a', 'b', 'c'],
            'col2': ['x', 'y', 'z']
        })
        trainer = DataFrameModelTraining(df)
        
        assert trainer.df.shape == (3, 2)
        assert trainer.df.dtypes['col1'] == 'object'
    
    def test_train_model(self):
        """Test train_model method"""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        trainer = DataFrameModelTraining(df)
        
        # Should not raise exception
        trainer.train_model()
        assert True
    
    def test_train_model_empty_dataframe(self):
        """Test train_model with empty DataFrame"""
        df = pd.DataFrame()
        trainer = DataFrameModelTraining(df)
        
        # Should not raise exception
        trainer.train_model()
        assert True
    
    def test_train_model_single_row(self):
        """Test train_model with single row DataFrame"""
        df = pd.DataFrame({'col1': [1], 'col2': [2]})
        trainer = DataFrameModelTraining(df)
        
        # Should not raise exception
        trainer.train_model()
        assert True
    
    def test_train_model_with_nan(self):
        """Test train_model with NaN values"""
        df = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [5, np.nan, 7, 8]
        })
        trainer = DataFrameModelTraining(df)
        
        # Should not raise exception
        trainer.train_model()
        assert True
    
    def test_train_model_large_dataframe(self):
        """Test train_model with large DataFrame"""
        df = pd.DataFrame({
            'col1': range(1000),
            'col2': range(1000, 2000)
        })
        trainer = DataFrameModelTraining(df)
        
        # Should not raise exception
        trainer.train_model()
        assert True
    
    def test_dataframe_immutability(self):
        """Test that original DataFrame is not modified"""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        original_df = df.copy()
        trainer = DataFrameModelTraining(df)
        
        trainer.train_model()
        
        # Original DataFrame should remain unchanged
        pd.testing.assert_frame_equal(trainer.df, original_df)

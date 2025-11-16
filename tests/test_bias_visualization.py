"""
Test cases for bias_visualization.py
"""

import pytest
from unittest.mock import Mock, patch
from src.bias_visualization import BiasVisualizer
from src.bias_detection import BiasReport, SliceMetrics


class TestBiasVisualizer:
    """Test cases for BiasVisualizer class"""
    
    @patch('os.makedirs')
    def test_init(self, mock_makedirs):
        """Test initialization"""
        visualizer = BiasVisualizer(output_dir="test_output")
        
        assert visualizer.output_dir == "test_output"
        mock_makedirs.assert_called_once()
    
    def test_generate_slice_comparison_chart(self):
        """Test generating slice comparison chart"""
        report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[
                SliceMetrics("slice1", "Popularity", "High", 100, 0.5, 0.6, 3.5, 3.4, 0.1, 0.2),
                SliceMetrics("slice2", "Popularity", "Low", 200, 0.6, 0.7, 3.6, 3.5, 0.1, 0.2)
            ],
            disparity_analysis={},
            recommendations=[]
        )
        
        visualizer = BiasVisualizer(output_dir="test_output")
        
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.close'):
                visualizer.generate_slice_comparison_chart(
                    report,
                    "Popularity",
                    "mae"
                )
                # Should not raise exception
                assert True
    
    def test_generate_disparity_heatmap(self):
        """Test generating disparity heatmap"""
        report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[
                SliceMetrics("slice1", "Popularity", "High", 100, 0.5, 0.6, 3.5, 3.4, 0.1, 0.2),
                SliceMetrics("slice2", "Popularity", "Low", 200, 0.6, 0.7, 3.6, 3.5, 0.1, 0.2)
            ],
            disparity_analysis={"summary": {}},
            recommendations=[]
        )
        
        visualizer = BiasVisualizer(output_dir="test_output")
        
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.close'):
                visualizer.generate_disparity_heatmap(report)
                # Should not raise exception
                assert True
    
    def test_create_fairness_scorecard(self):
        """Test creating fairness scorecard"""
        report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[],
            disparity_analysis={
                "detailed_disparities": [],
                "high_risk_slices": [],
                "summary": {}
            },
            recommendations=[]
        )
        
        visualizer = BiasVisualizer(output_dir="test_output")
        
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.close'):
                visualizer.create_fairness_scorecard(report)
                # Should not raise exception
                assert True
    
    def test_generate_all_visualizations(self):
        """Test generating all visualizations"""
        report = BiasReport(
            timestamp="2025-01-01",
            model_name="test-model",
            dataset="test",
            slice_metrics=[
                SliceMetrics("slice1", "Popularity", "High", 100, 0.5, 0.6, 3.5, 3.4, 0.1, 0.2)
            ],
            disparity_analysis={"summary": {}},
            recommendations=[]
        )
        
        visualizer = BiasVisualizer(output_dir="test_output")
        
        with patch.object(visualizer, 'create_fairness_scorecard'):
            with patch.object(visualizer, 'generate_disparity_heatmap'):
                with patch.object(visualizer, 'generate_slice_comparison_chart'):
                    visualizer.generate_all_visualizations(report)
                    # Should not raise exception
                    assert True


"""
Bias Visualization Module for Goodreads Recommendation System

This module generates visualizations to demonstrate model fairness:
1. Slice comparison charts (MAE/RMSE across groups)
2. Disparity heatmaps
3. Before/after mitigation comparisons
4. Fairness scorecards

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from datetime import datetime
from .bias_detection import BiasDetector, BiasReport


class BiasVisualizer:
    """
    Generate visualizations for bias analysis reports.
    """
    
    def __init__(self, output_dir: str = "../docs/bias_reports/visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        
        print(f"BiasVisualizer initialized")
        print(f"Output directory: {output_dir}\n")
    
    def generate_slice_comparison_chart(
        self,
        report: BiasReport,
        dimension: str,
        metric: str = 'mae',
        output_filename: Optional[str] = None
    ):
        """
        Generate bar chart comparing metrics across slices in a dimension.
        
        Args:
            report: BiasReport object
            dimension: Dimension to visualize (e.g., 'Popularity')
            metric: Metric to plot ('mae' or 'rmse')
            output_filename: Optional custom filename
        """
        # Filter metrics for this dimension
        dimension_metrics = [
            m for m in report.slice_metrics 
            if m.slice_dimension == dimension
        ]
        
        if not dimension_metrics:
            print(f"No metrics found for dimension: {dimension}")
            return
        
        # Prepare data
        slice_values = [m.slice_value for m in dimension_metrics]
        metric_values = [getattr(m, metric) for m in dimension_metrics]
        counts = [m.count for m in dimension_metrics]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bars with color gradient based on metric value
        colors = plt.cm.RdYlGn_r(
            [(v - min(metric_values)) / (max(metric_values) - min(metric_values) + 0.001) 
             for v in metric_values]
        )
        
        bars = ax.bar(slice_values, metric_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, val, count) in enumerate(zip(bars, metric_values, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}\n(n={count:,})',
                   ha='center', va='bottom', fontsize=9)
        
        # Add average line
        avg_value = np.mean(metric_values)
        ax.axhline(y=avg_value, color='red', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_value:.3f}')
        
        # Labels and title
        metric_name = metric.upper()
        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric_name}', fontsize=12, fontweight='bold')
        ax.set_title(f'{dimension}: {metric_name} Across Groups\n{report.model_name}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save
        if output_filename is None:
            output_filename = f"{report.model_name}_{dimension.replace(' ', '_')}_{metric}_comparison.png"
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    
    def generate_disparity_heatmap(
        self,
        report: BiasReport,
        output_filename: Optional[str] = None
    ):
        """
        Generate heatmap showing disparity across all dimensions.
        
        Args:
            report: BiasReport object
            output_filename: Optional custom filename
        """
        # Group metrics by dimension
        dimensions = {}
        for metric in report.slice_metrics:
            if metric.slice_dimension not in dimensions:
                dimensions[metric.slice_dimension] = []
            dimensions[metric.slice_dimension].append(metric)
        
        # Prepare data for heatmap
        data = []
        dimension_names = []
        
        for dim_name, dim_metrics in dimensions.items():
            if len(dim_metrics) < 2:
                continue
            
            maes = [m.mae for m in dim_metrics]
            rmses = [m.rmse for m in dim_metrics]
            
            mae_range = max(maes) - min(maes)
            rmse_range = max(rmses) - min(rmses)
            mae_cv = np.std(maes) / np.mean(maes) if np.mean(maes) > 0 else 0
            rmse_cv = np.std(rmses) / np.mean(rmses) if np.mean(rmses) > 0 else 0
            
            data.append({
                'Dimension': dim_name,
                'MAE Range': mae_range,
                'RMSE Range': rmse_range,
                'MAE CV': mae_cv,
                'RMSE CV': rmse_cv,
                'Num Slices': len(dim_metrics)
            })
            dimension_names.append(dim_name)
        
        df = pd.DataFrame(data)
        df = df.set_index('Dimension')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, len(dimension_names) * 0.6 + 2))
        
        # Select columns for heatmap
        heatmap_data = df[['MAE Range', 'MAE CV', 'RMSE Range', 'RMSE CV']]
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Disparity Score'}, ax=ax,
                   linewidths=1, linecolor='white')
        
        ax.set_title(f'Bias Disparity Heatmap: {report.model_name}\n'
                    f'Higher values indicate greater disparity',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Disparity Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dimension', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        if output_filename is None:
            output_filename = f"{report.model_name}_disparity_heatmap.png"
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    
    def generate_before_after_comparison(
        self,
        before_report: BiasReport,
        after_report: BiasReport,
        dimension: str,
        output_filename: Optional[str] = None
    ):
        """
        Generate before/after mitigation comparison chart.
        
        Args:
            before_report: BiasReport before mitigation
            after_report: BiasReport after mitigation
            dimension: Dimension to visualize
            output_filename: Optional custom filename
        """
        # Get metrics for this dimension
        before_metrics = [m for m in before_report.slice_metrics if m.slice_dimension == dimension]
        after_metrics = [m for m in after_report.slice_metrics if m.slice_dimension == dimension]
        
        if not before_metrics or not after_metrics:
            print(f"No metrics found for dimension: {dimension}")
            return
        
        # Prepare data
        slice_values = [m.slice_value for m in before_metrics]
        before_maes = [m.mae for m in before_metrics]
        after_maes = [m.mae for m in after_metrics]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Before chart
        x = np.arange(len(slice_values))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, before_maes, width, label='Before', 
                       color='#e74c3c', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x + width/2, after_maes, width, label='After',
                       color='#2ecc71', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Group', fontsize=11, fontweight='bold')
        ax1.set_ylabel('MAE', fontsize=11, fontweight='bold')
        ax1.set_title(f'Before vs After Mitigation\n{dimension}',
                     fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(slice_values, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Improvement chart (percentage change)
        improvements = [((b - a) / b * 100) if b > 0 else 0 
                       for b, a in zip(before_maes, after_maes)]
        
        colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        bars = ax2.bar(slice_values, improvements, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            label_pos = height if height > 0 else height
            ax2.text(bar.get_x() + bar.get_width()/2., label_pos,
                    f'{imp:+.1f}%', ha='center', 
                    va='bottom' if imp > 0 else 'top', fontsize=9)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Group', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
        ax2.set_title('MAE Improvement by Group\n(Positive = Better)',
                     fontsize=12, fontweight='bold')
        ax2.set_xticklabels(slice_values, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        if output_filename is None:
            output_filename = f"{before_report.model_name}_{dimension.replace(' ', '_')}_before_after.png"
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    
    def create_fairness_scorecard(
        self,
        report: BiasReport,
        output_filename: Optional[str] = None
    ):
        """
        Create a comprehensive fairness scorecard visualization.
        
        Args:
            report: BiasReport object
            output_filename: Optional custom filename
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Overall Fairness Score
        ax1 = fig.add_subplot(gs[0, :])
        
        # Calculate overall fairness score
        disparities = report.disparity_analysis.get('detailed_disparities', [])
        high_severity = sum(1 for d in disparities if d['severity'] == 'high')
        medium_severity = sum(1 for d in disparities if d['severity'] == 'medium')
        
        # Score: 100 - (high*20 + medium*10)
        fairness_score = max(0, 100 - (high_severity * 20 + medium_severity * 10))
        
        # Gauge chart
        colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
        wedges = [25, 25, 25, 25]
        
        wedge_colors, start_angle = [], 180
        for i, wedge in enumerate(wedges):
            wedge_colors.append(colors[i])
        
        ax1.pie(wedges, colors=wedge_colors, startangle=start_angle,
               counterclock=False, wedgeprops=dict(width=0.3))
        
        # Add needle
        theta = 180 - (fairness_score / 100) * 180
        ax1.arrow(0, 0, 0.5 * np.cos(np.radians(theta)), 
                 0.5 * np.sin(np.radians(theta)),
                 head_width=0.1, head_length=0.1, fc='black', ec='black', linewidth=2)
        
        # Add score text
        ax1.text(0, -0.3, f'{fairness_score:.0f}', 
                ha='center', va='center', fontsize=36, fontweight='bold')
        ax1.text(0, -0.5, 'Fairness Score', 
                ha='center', va='center', fontsize=14)
        
        ax1.set_title(f'Fairness Scorecard: {report.model_name}',
                     fontsize=16, fontweight='bold', pad=20)
        ax1.axis('equal')
        
        # 2. Disparity Summary
        ax2 = fig.add_subplot(gs[1, 0])
        
        summary_data = report.disparity_analysis.get('summary', {})
        if summary_data:
            dimensions = list(summary_data.keys())[:6]  # Top 6
            mae_cvs = [summary_data[d].get('mae_coefficient_of_variation', 0) for d in dimensions]
            
            colors_bar = ['#e74c3c' if cv > 0.25 else '#f39c12' if cv > 0.15 else '#2ecc71' 
                         for cv in mae_cvs]
            
            bars = ax2.barh(dimensions, mae_cvs, color=colors_bar, alpha=0.8, edgecolor='black')
            
            for bar, cv in zip(bars, mae_cvs):
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2,
                        f' {cv:.3f}', ha='left', va='center', fontsize=9)
            
            ax2.axvline(x=0.15, color='orange', linestyle='--', label='Medium threshold')
            ax2.axvline(x=0.25, color='red', linestyle='--', label='High threshold')
            ax2.set_xlabel('MAE Coefficient of Variation', fontweight='bold')
            ax2.set_title('Disparity by Dimension', fontweight='bold')
            ax2.legend(fontsize=8)
            ax2.grid(axis='x', alpha=0.3)
        
        # 3. High-Risk Slices
        ax3 = fig.add_subplot(gs[1, 1])
        
        high_risk = report.disparity_analysis.get('high_risk_slices', [])[:6]
        if high_risk:
            slice_names = [s['slice'].split('=')[1] if '=' in s['slice'] else s['slice'] 
                          for s in high_risk]
            maes = [s['mae'] for s in high_risk]
            
            bars = ax3.barh(slice_names, maes, color='#e74c3c', alpha=0.8, edgecolor='black')
            
            for bar, mae in zip(bars, maes):
                width = bar.get_width()
                ax3.text(width, bar.get_y() + bar.get_height()/2,
                        f' {mae:.3f}', ha='left', va='center', fontsize=9)
            
            ax3.set_xlabel('MAE', fontweight='bold')
            ax3.set_title('High-Risk Slices (Top 6)', fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
        
        # 4. Recommendations
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        recommendations = report.recommendations[:5]  # Top 5
        rec_text = "Key Recommendations:\n\n"
        for i, rec in enumerate(recommendations, 1):
            # Clean up emojis for text display
            clean_rec = rec.replace('⚠️', '[!]').replace('⚡', '[*]').replace('🎯', '[>]').replace('📊', '[-]')
            rec_text += f"{i}. {clean_rec}\n"
        
        ax4.text(0.05, 0.95, rec_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Save
        if output_filename is None:
            output_filename = f"{report.model_name}_fairness_scorecard.png"
        
        filepath = os.path.join(self.output_dir, output_filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    
    def generate_all_visualizations(self, report: BiasReport):
        """
        Generate all visualizations for a bias report.
        
        Args:
            report: BiasReport object
        """
        print(f"\n{'='*80}")
        print(f"GENERATING VISUALIZATIONS: {report.model_name}")
        print(f"{'='*80}\n")
        
        # 1. Fairness scorecard
        print("1. Creating fairness scorecard...")
        self.create_fairness_scorecard(report)
        
        # 2. Disparity heatmap
        print("2. Creating disparity heatmap...")
        self.generate_disparity_heatmap(report)
        
        # 3. Slice comparisons for each dimension
        print("3. Creating slice comparison charts...")
        dimensions = list(set(m.slice_dimension for m in report.slice_metrics))
        for i, dimension in enumerate(dimensions, 1):
            print(f"   {i}/{len(dimensions)}: {dimension}")
            self.generate_slice_comparison_chart(report, dimension, metric='mae')
        
        print(f"\nAll visualizations saved to: {self.output_dir}\n")
        print(f"{'='*80}\n")


def main():
    """Generate visualizations from existing bias report."""
    import sys
    
    visualizer = BiasVisualizer()
    
    # Load most recent report
    report_dir = "../docs/bias_reports"
    report_files = [f for f in os.listdir(report_dir) if f.endswith('_detection_report.json')]
    
    if not report_files:
        print("No bias reports found. Run bias detection first:")
        print("  python bias_detection.py")
        sys.exit(1)
    
    # Load the most recent report
    latest_report = sorted(report_files)[-1]
    report_path = os.path.join(report_dir, latest_report)
    
    print(f"Loading report: {report_path}")
    
    with open(report_path, 'r') as f:
        report_data = json.load(f)
    
    # Reconstruct BiasReport (simplified - just for visualization)
    from .bias_detection import SliceMetrics, BiasReport
    
    slice_metrics = [
        SliceMetrics(**m) for m in report_data['slice_metrics']
    ]
    
    report = BiasReport(
        timestamp=report_data['timestamp'],
        model_name=report_data['model_name'],
        dataset=report_data['dataset'],
        slice_metrics=slice_metrics,
        disparity_analysis=report_data['disparity_analysis'],
        recommendations=report_data['recommendations']
    )
    
    # Generate all visualizations
    visualizer.generate_all_visualizations(report)


if __name__ == "__main__":
    main()
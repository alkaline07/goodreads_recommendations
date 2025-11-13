"""
Model Selection Module - Based on Validation Performance AND Bias Analysis

This module ensures final model selection considers BOTH:
1. Validation performance (MAE, RMSE)
2. Bias/fairness metrics (disparity scores)

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from bias_detection import BiasDetector
from bias_visualization import BiasVisualizer


@dataclass
class ModelCandidate:
    """A model candidate for selection."""
    model_name: str
    predictions_table: str
    validation_mae: float
    validation_rmse: float
    performance_score: float
    fairness_score: float
    bias_disparities: int
    high_severity_disparities: int
    combined_score: float
    recommendation: str


@dataclass
class ModelSelectionReport:
    """Complete model selection report."""
    timestamp: str
    candidates: List[ModelCandidate]
    selected_model: ModelCandidate
    selection_criteria: Dict
    trade_offs: Dict
    rationale: str


class ModelSelector:
    """
    Select the best model considering both performance and fairness.
    """
    
    def __init__(
        self,
        performance_weight: float = 0.6,
        fairness_weight: float = 0.4,
        min_fairness_threshold: float = 50.0
    ):
        """
        Initialize model selector.
        
        Args:
            performance_weight: Weight for performance metrics (0-1)
            fairness_weight: Weight for fairness metrics (0-1)
            min_fairness_threshold: Minimum acceptable fairness score (0-100)
        """
        if abs(performance_weight + fairness_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        
        self.performance_weight = performance_weight
        self.fairness_weight = fairness_weight
        self.min_fairness_threshold = min_fairness_threshold
        
        self.detector = BiasDetector()
        self.visualizer = BiasVisualizer(output_dir="data/bias_reports/model_selection")
        
        print(f"ModelSelector initialized")
        print(f"  Performance weight: {performance_weight}")
        print(f"  Fairness weight: {fairness_weight}")
        print(f"  Min fairness threshold: {min_fairness_threshold}\n")
    
    def calculate_fairness_score(self, bias_report) -> float:
        """
        Calculate fairness score from bias report.
        
        Score calculation:
        - Start with 100
        - Subtract 20 points for each high-severity disparity
        - Subtract 10 points for each medium-severity disparity
        - Minimum score is 0
        
        Args:
            bias_report: BiasReport object
            
        Returns:
            Fairness score (0-100)
        """
        disparities = bias_report.disparity_analysis.get('detailed_disparities', [])
        
        high_severity = sum(1 for d in disparities if d['severity'] == 'high')
        medium_severity = sum(1 for d in disparities if d['severity'] == 'medium')
        
        score = 100 - (high_severity * 20 + medium_severity * 10)
        return max(0, score)
    
    def normalize_performance_score(
        self,
        mae: float,
        rmse: float,
        all_maes: List[float],
        all_rmses: List[float]
    ) -> float:
        """
        Normalize performance metrics to 0-100 scale (higher is better).
        
        Uses a hybrid approach:
        - With 3+ models: Uses min-max normalization for relative comparison
        - With 2 models: Uses percentage-based scoring from best model
        
        This prevents artificial extremes (100 vs 0) when comparing only 2 models.

        Args:
            mae: Model's MAE
            rmse: Model's RMSE
            all_maes: All models' MAEs for normalization
            all_rmses: All models' RMSEs for normalization
            
        Returns:
            Normalized performance score (0-100)
        """
        
        mae_min, mae_max = min(all_maes), max(all_maes)
        rmse_min, rmse_max = min(all_rmses), max(all_rmses)

        # If only 2 models, use percentage-based scoring to show actual gaps
        if len(all_maes) == 2:
            # Calculate percentage difference from best model
            mae_pct_diff = ((mae - mae_min) / mae_min) * 100 if mae_min > 0 else 0
            rmse_pct_diff = ((rmse - rmse_min) / rmse_min) * 100 if rmse_min > 0 else 0
            
            # Score: 100 - (percentage worse)
            # Cap at 0 minimum, allow models within 100% of best to have positive scores
            mae_score = max(0, 100 - mae_pct_diff)
            rmse_score = max(0, 100 - rmse_pct_diff)
            
            performance_score = (mae_score + rmse_score) / 2
            
            return performance_score
        
        # With 3+ models, use traditional min-max normalization
        else:

            # Avoid division by zero
            mae_range = mae_max - mae_min if mae_max > mae_min else 1.0
            rmse_range = rmse_max - rmse_min if rmse_max > rmse_min else 1.0
            
            # Normalize (lower is better, so subtract from 1)
            mae_norm = 1 - ((mae - mae_min) / mae_range)
            rmse_norm = 1 - ((rmse - rmse_min) / rmse_range)
            
            # Average and scale to 0-100
            performance_score = ((mae_norm + rmse_norm) / 2) * 100
            
            return performance_score
    
    def calculate_combined_score(
        self,
        performance_score: float,
        fairness_score: float
    ) -> float:
        """
        Calculate combined score from performance and fairness.
        
        Args:
            performance_score: Normalized performance score (0-100)
            fairness_score: Fairness score (0-100)
            
        Returns:
            Combined score (0-100)
        """
        combined = (
            self.performance_weight * performance_score +
            self.fairness_weight * fairness_score
        )
        return combined
    
    def evaluate_model(
        self,
        model_name: str,
        predictions_table: str
    ) -> Tuple[float, float, 'BiasReport']:
        """
        Evaluate a model on validation set.
        
        Args:
            model_name: Name of the model
            predictions_table: BigQuery table with predictions
            
        Returns:
            Tuple of (validation_mae, validation_rmse, bias_report)
        """
        print(f"\nEvaluating model: {model_name}")
        print(f"  Predictions table: {predictions_table}")
        
        # Get validation performance
        query = f"""
        SELECT
            AVG(ABS(actual_rating - predicted_rating)) as mae,
            SQRT(AVG(POWER(actual_rating - predicted_rating, 2))) as rmse
        FROM `{predictions_table}`
        WHERE actual_rating IS NOT NULL AND predicted_rating IS NOT NULL
        """
        
        try:
            stats = self.detector.client.query(query).to_dataframe(create_bqstorage_client=False)
            mae = float(stats['mae'].iloc[0])
            rmse = float(stats['rmse'].iloc[0])
        except Exception as e:
            print(f"  Error getting validation metrics: {e}")
            raise
        
        # Run bias detection
        bias_report = self.detector.detect_bias(
            predictions_table=predictions_table,
            model_name=model_name,
            dataset="test"
        )
        
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        return mae, rmse, bias_report
    
    def compare_models(
        self,
        model_candidates: List[Dict[str, str]]
    ) -> ModelSelectionReport:
        """
        Compare multiple models and select the best one.
        
        Args:
            model_candidates: List of dicts with 'model_name' and 'predictions_table'
            
        Returns:
            ModelSelectionReport with selected model and rationale
        """
        print("\n" + "="*80)
        print("MODEL SELECTION BASED ON PERFORMANCE + FAIRNESS")
        print("="*80)
        
        print(f"\nEvaluating {len(model_candidates)} models...")
        print(f"Criteria: {self.performance_weight*100:.0f}% Performance + "
             f"{self.fairness_weight*100:.0f}% Fairness")
        
        if len(model_candidates) == 2:
            print("\nNote: With 2 models, performance scores use percentage-based comparison")
            print("   This shows actual performance gaps rather than artificial extremes\n")
        else:
            print()
        
        # Evaluate all models
        candidates = []
        all_maes = []
        all_rmses = []
        
        for candidate_info in model_candidates:
            try:
                mae, rmse, bias_report = self.evaluate_model(
                    candidate_info['model_name'],
                    candidate_info['predictions_table']
                )
                
                fairness_score = self.calculate_fairness_score(bias_report)
                
                disparities = bias_report.disparity_analysis.get('detailed_disparities', [])
                high_severity = sum(1 for d in disparities if d['severity'] == 'high')
                
                candidate = {
                    'model_name': candidate_info['model_name'],
                    'predictions_table': candidate_info['predictions_table'],
                    'mae': mae,
                    'rmse': rmse,
                    'fairness_score': fairness_score,
                    'bias_report': bias_report,
                    'disparities_count': len(disparities),
                    'high_severity_count': high_severity
                }
                
                candidates.append(candidate)
                all_maes.append(mae)
                all_rmses.append(rmse)
                
            except Exception as e:
                print(f"Failed to evaluate {candidate_info['model_name']}: {e}")
                continue
        
        if not candidates:
            raise ValueError("No models could be evaluated successfully")
        
        # Calculate normalized performance scores and combined scores
        for candidate in candidates:
            perf_score = self.normalize_performance_score(
                candidate['mae'],
                candidate['rmse'],
                all_maes,
                all_rmses
            )
            
            combined_score = self.calculate_combined_score(
                perf_score,
                candidate['fairness_score']
            )
            
            candidate['performance_score'] = perf_score
            candidate['combined_score'] = combined_score

            # Log scoring details for transparency
            print(f"\n  {candidate['model_name']}:")
            print(f"    MAE: {candidate['mae']:.4f} | RMSE: {candidate['rmse']:.4f}")
            print(f"    Performance Score: {perf_score:.1f}/100")
            print(f"    Fairness Score: {candidate['fairness_score']}/100")
            print(f"    Combined Score: {combined_score:.1f}/100")
        
        # Sort by combined score (descending)
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Select best model (with fairness threshold check)
        selected = None
        for candidate in candidates:
            if candidate['fairness_score'] >= self.min_fairness_threshold:
                selected = candidate
                break
        
        # If no model meets fairness threshold, take best fairness score
        if selected is None:
            selected = max(candidates, key=lambda x: x['fairness_score'])
            print(f"\nWARNING: No model meets fairness threshold ({self.min_fairness_threshold})")
            print(f"   Selecting model with best fairness score instead")
        
        # Generate selection report
        report = self._generate_selection_report(candidates, selected)
        
        # Print summary
        self._print_selection_summary(report)
        
        # Save report
        report_path = self._save_selection_report(report)
        
        # Generate comparison visualizations
        self._generate_comparison_visualizations(candidates, selected)
        
        return report
    
    def _generate_selection_report(
        self,
        candidates: List[Dict],
        selected: Dict
    ) -> ModelSelectionReport:
        """Generate comprehensive selection report."""
        
        # Convert candidates to ModelCandidate objects
        candidate_objects = []
        for c in candidates:
            recommendation = "SELECTED" if c == selected else "NOT SELECTED"
            
            if c != selected:
                if c['fairness_score'] < self.min_fairness_threshold:
                    recommendation += " - Below fairness threshold"
                elif c['combined_score'] < selected['combined_score']:
                    recommendation += f" - Lower combined score ({c['combined_score']:.1f} vs {selected['combined_score']:.1f})"
            
            candidate_obj = ModelCandidate(
                model_name=c['model_name'],
                predictions_table=c['predictions_table'],
                validation_mae=c['mae'],
                validation_rmse=c['rmse'],
                performance_score=c['performance_score'],
                fairness_score=c['fairness_score'],
                bias_disparities=c['disparities_count'],
                high_severity_disparities=c['high_severity_count'],
                combined_score=c['combined_score'],
                recommendation=recommendation
            )
            candidate_objects.append(candidate_obj)
        
        # Selection criteria
        criteria = {
            'performance_weight': self.performance_weight,
            'fairness_weight': self.fairness_weight,
            'min_fairness_threshold': self.min_fairness_threshold,
            'scoring_formula': f"Combined Score = {self.performance_weight}×Performance + {self.fairness_weight}×Fairness"
        }
        
        # Trade-offs analysis
        best_performance = min(candidates, key=lambda x: x['mae'])
        best_fairness = max(candidates, key=lambda x: x['fairness_score'])
        
        trade_offs = {
            'best_performance_model': best_performance['model_name'],
            'best_performance_mae': best_performance['mae'],
            'best_fairness_model': best_fairness['model_name'],
            'best_fairness_score': best_fairness['fairness_score'],
            'selected_model_is_best_performance': (selected == best_performance),
            'selected_model_is_best_fairness': (selected == best_fairness),
            'performance_sacrifice': ((selected['mae'] - best_performance['mae']) / best_performance['mae'] * 100) if selected != best_performance else 0,
            'fairness_gain': (selected['fairness_score'] - best_performance['fairness_score']) if selected != best_performance else 0
        }
        
        # Rationale
        rationale_parts = []
        rationale_parts.append(f"Selected '{selected['model_name']}' with combined score of {selected['combined_score']:.1f}/100.")

        if selected == best_performance and selected == best_fairness:
            if selected['fairness_score'] >= 80:
                rationale_parts.append("This model achieves BOTH the best performance AND best fairness.")
            elif selected['fairness_score'] >= 70:
                rationale_parts.append("This model has the best performance and is the least biased among candidates.")
            else:
                rationale_parts.append("This model outperforms alternatives in both accuracy and fairness, though fairness could be improved.")
        elif selected == best_performance:
            rationale_parts.append("This model has the best validation performance.")
        elif selected == best_fairness:
            if selected['fairness_score'] >= 80:
                rationale_parts.append("This model has excellent fairness.")
            else:
                rationale_parts.append("This model has the best fairness score among candidates.")
        else:
            rationale_parts.append(f"This model balances performance and fairness optimally under the {self.performance_weight*100:.0f}/{self.fairness_weight*100:.0f} weighting.")
        
        if trade_offs['performance_sacrifice'] > 0:
            rationale_parts.append(f"Accepts {trade_offs['performance_sacrifice']:.1f}% higher MAE for {trade_offs['fairness_gain']:.1f} point fairness improvement.")
        
        # Enhanced warning with severity context
        if selected['high_severity_count'] > 0:
            if selected['fairness_score'] < 70:
                rationale_parts.append(f"⚠️ CAUTION: Model has {selected['high_severity_count']} high-severity bias disparities and low fairness ({selected['fairness_score']}/100). Mitigation strongly recommended before deployment.")
            else:
                rationale_parts.append(f"⚠️ NOTE: Model has {selected['high_severity_count']} high-severity bias disparities. Consider mitigation before deployment.")
        elif selected['disparities_count'] > 0:
            rationale_parts.append(f"Model has {selected['disparities_count']} moderate bias disparities. Monitor for fairness in production.")
        
        rationale = " ".join(rationale_parts)
        
        selected_obj = next(c for c in candidate_objects if c.model_name == selected['model_name'])
        
        return ModelSelectionReport(
            timestamp=datetime.now().isoformat(),
            candidates=candidate_objects,
            selected_model=selected_obj,
            selection_criteria=criteria,
            trade_offs=trade_offs,
            rationale=rationale
        )
    
    def _print_selection_summary(self, report: ModelSelectionReport):
        """Print selection summary to console."""
        print("\n" + "="*80)
        print("MODEL SELECTION SUMMARY")
        print("="*80)
        
        print(f"\nSELECTED MODEL: {report.selected_model.model_name}")
        print(f"   Combined Score: {report.selected_model.combined_score:.1f}/100")
        print(f"   Performance Score: {report.selected_model.performance_score:.1f}/100")
        print(f"   Fairness Score: {report.selected_model.fairness_score:.1f}/100")
        print(f"   Validation MAE: {report.selected_model.validation_mae:.4f}")
        print(f"   Validation RMSE: {report.selected_model.validation_rmse:.4f}")
        
        print(f"\nALL CANDIDATES:")
        print(f"{'Model':<30} {'MAE':<10} {'Perf':<8} {'Fair':<8} {'Combined':<10} {'Status'}")
        print("-" * 85)
        
        for candidate in report.candidates:
            status = "SELECTED" if candidate == report.selected_model else "  "
            print(f"{candidate.model_name:<30} {candidate.validation_mae:<10.4f} "
                  f"{candidate.performance_score:<8.1f} {candidate.fairness_score:<8.1f} "
                  f"{candidate.combined_score:<10.1f} {status}")
        
        print(f"\nRATIONALE:")
        print(f"   {report.rationale}")
        
        print(f"\n⚖️  TRADE-OFFS:")
        if not report.trade_offs['selected_model_is_best_performance']:
            print(f"   Performance sacrifice: {report.trade_offs['performance_sacrifice']:.1f}% higher MAE")
            print(f"   Fairness gain: +{report.trade_offs['fairness_gain']:.1f} fairness points")
        else:
            print(f"   No trade-off: Selected model is also the best performer")
        
        print("\n" + "="*80 + "\n")
    
    def _save_selection_report(self, report: ModelSelectionReport) -> str:
        """Save selection report to JSON file."""
        report_dict = {
            'timestamp': report.timestamp,
            'selected_model': asdict(report.selected_model),
            'all_candidates': [asdict(c) for c in report.candidates],
            'selection_criteria': report.selection_criteria,
            'trade_offs': report.trade_offs,
            'rationale': report.rationale
        }
        
        output_path = "data/bias_reports/model_selection_report.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"Selection report saved: {output_path}")
        
        return output_path
    
    def _generate_comparison_visualizations(
        self,
        candidates: List[Dict],
        selected: Dict
    ):
        """Generate comparison visualizations."""
        import matplotlib.pyplot as plt
        
        print("\nGenerating comparison visualizations...")
        
        # Create comparison chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        names = [c['model_name'] for c in candidates]
        maes = [c['mae'] for c in candidates]
        fairness = [c['fairness_score'] for c in candidates]
        combined = [c['combined_score'] for c in candidates]
        
        # Highlight selected model
        colors = ['#2ecc71' if c == selected else '#95a5a6' for c in candidates]
        
        # 1. MAE comparison
        bars1 = ax1.barh(names, maes, color=colors, edgecolor='black')
        ax1.set_xlabel('MAE (lower is better)', fontweight='bold')
        ax1.set_title('Validation Performance', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Fairness comparison
        bars2 = ax2.barh(names, fairness, color=colors, edgecolor='black')
        ax2.axvline(x=self.min_fairness_threshold, color='red', linestyle='--', 
                   label=f'Min threshold: {self.min_fairness_threshold}')
        ax2.set_xlabel('Fairness Score (higher is better)', fontweight='bold')
        ax2.set_title('Fairness Score', fontweight='bold')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Combined score
        bars3 = ax3.barh(names, combined, color=colors, edgecolor='black')
        ax3.set_xlabel('Combined Score (higher is better)', fontweight='bold')
        ax3.set_title(f'Combined Score ({self.performance_weight*100:.0f}% Perf + {self.fairness_weight*100:.0f}% Fair)', 
                     fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Performance vs Fairness scatter
        ax4.scatter(maes, fairness, s=200, c=colors, edgecolors='black', linewidth=2)
        for i, name in enumerate(names):
            ax4.annotate(name, (maes[i], fairness[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.axhline(y=self.min_fairness_threshold, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('MAE (lower is better)', fontweight='bold')
        ax4.set_ylabel('Fairness Score (higher is better)', fontweight='bold')
        ax4.set_title('Performance vs Fairness Trade-off', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = "data/bias_reports/model_selection/model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def main():
    """Example usage."""
    selector = ModelSelector(
        performance_weight=0.6,
        fairness_weight=0.4,
        min_fairness_threshold=60.0
    )
    
    # Define model candidates
    candidates = [
        {
            'model_name': 'boosted_tree_regressor',
            'predictions_table': f'{selector.detector.project_id}.books.boosted_tree_rating_predictions'
        },
        {
            'model_name': 'matrix_factorization',
            'predictions_table': f'{selector.detector.project_id}.books.matrix_factorization_rating_predictions'
        },
        # {
        #      'model_name': 'automl_regressor',
        #      'predictions_table': f'{selector.detector.project_id}.books.automl_rating_predictions'
        # },
    ]
    
    # Compare and select
    report = selector.compare_models(candidates)
    
    print(f"\nFINAL SELECTION: {report.selected_model.model_name}")
    print(f"   Use table: {report.selected_model.predictions_table}")


if __name__ == "__main__":
    main()
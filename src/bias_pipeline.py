"""
Integrated Bias Detection and Mitigation Pipeline for Goodreads Recommendation System

This module orchestrates the complete bias detection and mitigation workflow:
1. Detects bias across multiple demographic slices
2. Applies appropriate mitigation techniques
3. Validates mitigation effectiveness
4. Generates comprehensive reports

Author: Goodreads Recommendation Team
Date: 2025
"""

import os
from datetime import datetime
from typing import List, Dict, Optional
import json
import argparse
from bias_detection import BiasDetector, BiasReport
from bias_mitigation import BiasMitigator, MitigationResult
from bias_visualization import BiasVisualizer
from model_selector import ModelSelector


class BiasAuditPipeline:
    """
    Complete pipeline for bias auditing and mitigation.
    """
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            project_id: GCP project ID
        """
        self.detector = BiasDetector(project_id=project_id)
        self.mitigator = BiasMitigator(project_id=project_id)
        self.visualizer = BiasVisualizer()
        self.model_selector = ModelSelector()
        self.project_id = self.detector.project_id
        self.dataset_id = "books"
        
        print(f"BiasAuditPipeline initialized for project: {self.project_id}")
    
    def run_full_audit(
        self,
        model_name: str,
        predictions_table: str,
        apply_mitigation: bool = True,
        mitigation_techniques: List[str] = None,
        generate_visualizations: bool = True,
        enable_model_selection: bool = False
    ) -> Dict:
        """
        Run a complete bias audit on a model.
        
        Args:
            model_name: Name of the model
            predictions_table: BigQuery table with predictions
            apply_mitigation: Whether to apply mitigation
            mitigation_techniques: List of techniques to apply
                ['shrinkage', 'threshold_adjustment', 'reweighting']
            generate_visualizations: Whether to generate visual reports (default: True)
            enable_model_selection: Whether to compare with other models (default: False)
        
        Returns:
            Dictionary with audit results
        """
        print("\n" + "="*80)
        print(f"BIAS AUDIT PIPELINE: {model_name}")
        print("="*80 + "\n")
        
        audit_results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'predictions_table': predictions_table,
            'detection_report': None,
            'visualizations_generated': False,
            'mitigation_results': [],
            'final_validation': None,
            'model_selection': None
        }
        
        # Step 1: Detect Bias
        print("\n[STEP 1/4] Running Bias Detection...")
        detection_report = self.detector.detect_bias(
            predictions_table=predictions_table,
            model_name=model_name,
            dataset="test"
        )
        audit_results['detection_report'] = detection_report
        
        # Save detection report
        detection_path = f"data/bias_reports/{model_name}_detection_report.json"
        self.detector.save_report(detection_report, detection_path)
        
        # Save to BigQuery
        metrics_table = f"{self.project_id}.{self.dataset_id}.bias_metrics_{model_name}"
        self.detector.create_bias_metrics_table(detection_report, metrics_table)
        
        # Step 1.5: Generate Visualizations (if requested)
        if generate_visualizations:
            print("\n[STEP 1.5/5] Generating Visualizations...")
            try:
                self.visualizer.generate_all_visualizations(detection_report)
                audit_results['visualizations_generated'] = True
            except Exception as e:
                print(f"⚠️ Warning: Could not generate visualizations: {e}")
                audit_results['visualizations_generated'] = False
        
        # Step 2: Apply Mitigation (if requested and bias detected)
        if apply_mitigation and detection_report.disparity_analysis['detailed_disparities']:
            print("\n[STEP 2/5] Applying Bias Mitigation...")
            
            if mitigation_techniques is None:
                mitigation_techniques = ['threshold_adjustment']
            
            for technique in mitigation_techniques:
                if technique == 'threshold_adjustment':
                    mitigation_result = self._apply_threshold_mitigation(
                        predictions_table,
                        model_name,
                        detection_report
                    )
                    if mitigation_result:
                        audit_results['mitigation_results'].append(mitigation_result)
                    
                elif technique == 'shrinkage':
                    features_table = f"{self.project_id}.{self.dataset_id}.goodreads_features"
                    mitigation_result = self._apply_shrinkage_mitigation(
                        features_table,
                        model_name,
                        detection_report
                    )
                    if mitigation_result:
                        audit_results['mitigation_results'].append(mitigation_result)
        else:
            print("\n[STEP 2/5] Skipping mitigation (no significant bias detected or not requested)")
        
        # Step 3: Validate Mitigation
        if audit_results['mitigation_results']:
            print("\n[STEP 3/5] Validating Mitigation Effectiveness...")
            validation = self._validate_mitigation(
                audit_results['mitigation_results'],
                model_name
            )
            audit_results['final_validation'] = validation
        else:
            print("\n[STEP 3/5] Skipping validation (no mitigation applied)")
        
        # Step 4: Generate Comprehensive Report
        print("\n[STEP 4/5] Generating Comprehensive Audit Report...")
        report_path = self._generate_comprehensive_report(audit_results, model_name)
        audit_results['report_path'] = report_path
        
        # Step 5: Model Selection (if enabled and not already in selection mode)
        if enable_model_selection and not audit_results.get('skip_selection'):
            print("\n[STEP 5/5] Model Selection Across All Candidates...")
            print("Note: This step is handled by run_model_selection() instead.")
            print("      Call run_model_selection() to compare multiple models.")
        else:
            print("\n[STEP 5/5] Skipping model selection (disabled or handled separately)")
        
        print("\n" + "="*80)
        print("BIAS AUDIT COMPLETE")
        print("="*80)
        print(f"\nFull audit report: {report_path}")
        if audit_results['visualizations_generated']:
            print("Visualizations: data/bias_reports/visualizations/")
        
        return audit_results
    
    def _apply_threshold_mitigation(
        self,
        predictions_table: str,
        model_name: str,
        detection_report: BiasReport
    ) -> MitigationResult:
        """Apply threshold adjustment mitigation to high-disparity dimensions."""
        print("\n  Applying threshold adjustments...")
        
        # Find the dimension with highest disparity
        disparities = detection_report.disparity_analysis['detailed_disparities']
        if not disparities:
            return None
        
        target_dim = max(disparities, key=lambda x: x['mae_cv'])
        
        # Get the slice expression for this dimension
        slice_definitions = self.detector.get_slice_definitions()
        slice_expr = None
        for dim_name, expr, label in slice_definitions:
            if dim_name == target_dim['dimension']:
                slice_expr = expr
                break
        
        if slice_expr is None:
            print(f"  Could not find slice expression for {target_dim['dimension']}")
            return None
        
        output_table = f"{self.project_id}.{self.dataset_id}.{model_name}_predictions_mitigated"
        
        result = self.mitigator.apply_threshold_adjustments(
            predictions_table=predictions_table,
            output_table=output_table,
            slice_dimension=target_dim['dimension'],
            slice_expression=slice_expr
        )
        
        # Save mitigation report
        report_path = f"data/bias_reports/{model_name}_threshold_mitigation.json"
        self.mitigator.save_mitigation_report(result, report_path)
        
        return result
    
    def _apply_shrinkage_mitigation(
        self,
        features_table: str,
        model_name: str,
        detection_report: BiasReport
    ) -> MitigationResult:
        """Apply shrinkage mitigation to reduce group bias."""
        print("\n  Applying shrinkage mitigation...")
        
        disparities = detection_report.disparity_analysis['detailed_disparities']
        if not disparities:
            return None
        
        target_dim = max(disparities, key=lambda x: x['mae_cv'])
        
        # Get the slice expression
        slice_definitions = self.detector.get_slice_definitions()
        slice_expr = None
        for dim_name, expr, label in slice_definitions:
            if dim_name == target_dim['dimension']:
                slice_expr = expr
                break
        
        if slice_expr is None:
            return None
        
        output_table = f"{self.project_id}.{self.dataset_id}.{model_name}_features_debiased"
        
        result = self.mitigator.apply_shrinkage_mitigation(
            features_table=features_table,
            output_table=output_table,
            slice_dimension=target_dim['dimension'],
            slice_expression=slice_expr,
            lambda_shrinkage=0.5
        )
        
        report_path = f"data/bias_reports/{model_name}_shrinkage_mitigation.json"
        self.mitigator.save_mitigation_report(result, report_path)
        
        return result
    
    def _validate_mitigation(
        self,
        mitigation_results: List[MitigationResult],
        model_name: str
    ) -> Dict:
        """Validate the effectiveness of applied mitigation techniques."""
        validation = {
            'timestamp': datetime.now().isoformat(),
            'techniques_applied': [r.technique for r in mitigation_results],
            'effectiveness': {}
        }
        
        for result in mitigation_results:
            if not result:
                continue
            if result.improvement_pct:
                validation['effectiveness'][result.technique] = {
                    'improvements': result.improvement_pct,
                    'output_table': result.output_table
                }
                
                # Re-run bias detection on mitigated predictions
                if result.technique == 'threshold_adjustment':
                    print(f"\n  Re-running bias detection on mitigated predictions...")
                    post_mitigation_report = self.detector.detect_bias(
                        predictions_table=result.output_table,
                        model_name=f"{model_name}_mitigated",
                        dataset="test"
                    )
                    
                    validation['effectiveness'][result.technique]['post_mitigation_disparities'] = \
                        post_mitigation_report.disparity_analysis['detailed_disparities']
        
        return validation
    
    def _generate_comprehensive_report(
        self,
        audit_results: Dict,
        model_name: str
    ) -> str:
        """Generate a comprehensive audit report."""
        report = {
            'audit_metadata': {
                'model_name': audit_results['model_name'],
                'timestamp': audit_results['timestamp'],
                'predictions_table': audit_results['predictions_table']
            },
            'bias_detection': {
                'timestamp': audit_results['detection_report'].timestamp,
                'total_slices_analyzed': len(audit_results['detection_report'].slice_metrics),
                'disparities_found': len(audit_results['detection_report'].disparity_analysis['detailed_disparities']),
                'high_risk_slices': len(audit_results['detection_report'].disparity_analysis['high_risk_slices']),
                'recommendations': audit_results['detection_report'].recommendations
            },
            'mitigation_applied': {
                'techniques': [r.technique for r in audit_results['mitigation_results']],
                'results': [
                    {
                        'technique': r.technique,
                        'output_table': r.output_table,
                        'improvements': r.improvement_pct
                    }
                    for r in audit_results['mitigation_results']
                ]
            } if audit_results['mitigation_results'] else None,
            'validation': audit_results.get('final_validation'),
            'executive_summary': self._generate_executive_summary(audit_results)
        }
        
        output_path = f"data/bias_reports/{model_name}_comprehensive_audit.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Comprehensive audit report saved to: {output_path}")
        
        return output_path
    
    def _generate_executive_summary(self, audit_results: Dict) -> Dict:
        """Generate an executive summary of the audit."""
        detection = audit_results['detection_report']
        
        summary = {
            'bias_detected': len(detection.disparity_analysis['detailed_disparities']) > 0,
            'severity': 'high' if any(d['severity'] == 'high' for d in detection.disparity_analysis['detailed_disparities']) else 'medium' if detection.disparity_analysis['detailed_disparities'] else 'none',
            'dimensions_with_bias': [d['dimension'] for d in detection.disparity_analysis['detailed_disparities']],
            'mitigation_applied': len(audit_results['mitigation_results']) > 0,
            'overall_status': 'PASS' if len(detection.disparity_analysis['detailed_disparities']) == 0 else 'MITIGATED' if audit_results['mitigation_results'] else 'NEEDS_ATTENTION'
        }
        
        return summary


    def run_model_selection(
        self,
        model_candidates: List[Dict[str, str]],
        performance_weight: float = 0.6,
        fairness_weight: float = 0.4,
        min_fairness_threshold: float = 60.0,
        generate_visualizations: bool = True
    ) -> Dict:
        """
        Compare multiple models and select the best one based on performance + fairness.
        
        Args:
            model_candidates: List of dicts with 'model_name' and 'predictions_table'
            performance_weight: Weight for performance metrics (0-1)
            fairness_weight: Weight for fairness metrics (0-1)
            min_fairness_threshold: Minimum acceptable fairness score
            generate_visualizations: Whether to generate visualizations for each model
            
        Returns:
            Dictionary with selection results
        """
        print("\n" + "="*80)
        print("MODEL SELECTION PIPELINE (PERFORMANCE + FAIRNESS)")
        print("="*80)
        print(f"\nComparing {len(model_candidates)} models...")
        print(f"Weights: {performance_weight*100:.0f}% Performance + {fairness_weight*100:.0f}% Fairness")
        print(f"Minimum Fairness Threshold: {min_fairness_threshold}")
        
        # Run full audit for each model (without individual selection)
        all_audits = []
        for i, candidate in enumerate(model_candidates, 1):
            print(f"\n\n{'#'*80}")
            print(f"# AUDITING MODEL {i}/{len(model_candidates)}: {candidate['model_name']}")
            print(f"{'#'*80}\n")
            
            try:
                audit_result = self.run_full_audit(
                    model_name=candidate['model_name'],
                    predictions_table=candidate['predictions_table'],
                    apply_mitigation=False,  # Don't apply mitigation yet
                    generate_visualizations=generate_visualizations,
                    enable_model_selection=False  # Disable individual selection
                )
                audit_result['skip_selection'] = True  # Mark that we're in selection mode
                all_audits.append(audit_result)
            except Exception as e:
                print(f"✗ Error auditing {candidate['model_name']}: {e}")
                continue
        
        if not all_audits:
            print("\n✗ No models could be audited successfully")
            return {'error': 'No successful audits'}
        
        # Now run model selection
        print(f"\n\n{'='*80}")
        print("RUNNING MODEL SELECTION")
        print(f"{'='*80}\n")
        
        # Update model selector with custom weights
        self.model_selector = ModelSelector(
            performance_weight=performance_weight,
            fairness_weight=fairness_weight,
            min_fairness_threshold=min_fairness_threshold
        )
        
        selection_report = self.model_selector.compare_models(model_candidates)
        
        # Final summary
        print(f"\n\n{'='*80}")
        print("MODEL SELECTION COMPLETE")
        print(f"{'='*80}\n")
        print(f"🎯 SELECTED MODEL: {selection_report.selected_model.model_name}")
        print(f"   Validation MAE: {selection_report.selected_model.validation_mae:.4f}")
        print(f"   Fairness Score: {selection_report.selected_model.fairness_score:.1f}/100")
        print(f"   Combined Score: {selection_report.selected_model.combined_score:.1f}/100")
        print(f"\n📋 RATIONALE:")
        print(f"   {selection_report.rationale}")
        print(f"\n📊 Reports:")
        print(f"   Model Selection: data/bias_reports/model_selection_report.json")
        print(f"   Visualizations: data/bias_reports/model_selection/")
        
        return {
            'selection_report': selection_report,
            'all_audits': all_audits,
            'selected_model': selection_report.selected_model.model_name,
            'selected_table': selection_report.selected_model.predictions_table
        }


def run_bias_audit_for_all_models(with_model_selection: bool = True):
    """
    Run bias audit for all trained models.
    
    Args:
        with_model_selection: If True, compare models and select best one (default: True)
    """
    pipeline = BiasAuditPipeline()
    
    models_to_audit = [
        {
            'model_name': 'boosted_tree_regressor',
            'predictions_table': f"{pipeline.project_id}.{pipeline.dataset_id}.boosted_tree_rating_predictions"
        },
        {
            'model_name': 'automl_regressor',
            'predictions_table': f"{pipeline.project_id}.{pipeline.dataset_id}.automl_rating_predictions"
        },
        {
            'model_name': 'matrix_factorization',
            'predictions_table': f"{pipeline.project_id}.{pipeline.dataset_id}.matrix_factorization_rating_predictions"
        }
    ]
    
    if with_model_selection:
        # Run integrated model selection pipeline
        print("Running integrated pipeline with MODEL SELECTION...\n")
        results = pipeline.run_model_selection(
            model_candidates=models_to_audit,
            performance_weight=0.6,
            fairness_weight=0.4,
            min_fairness_threshold=60.0,
            generate_visualizations=True
        )
        
        print(f"\n\n{'='*80}")
        print("COMPLETE PIPELINE FINISHED")
        print(f"{'='*80}")
        print(f"\n🎯 SELECTED MODEL: {results['selected_model']}")
        print(f"   Use this for production: {results['selected_table']}")
        
        return results
    
    else:
        # Run individual audits without selection
        print("Running individual audits WITHOUT model selection...\n")
        all_results = {}
        
        for model in models_to_audit:
            print(f"\n\n{'#'*80}")
            print(f"# AUDITING MODEL: {model['model_name']}")
            print(f"{'#'*80}\n")
            
            try:
                results = pipeline.run_full_audit(
                    model_name=model['model_name'],
                    predictions_table=model['predictions_table'],
                    apply_mitigation=True,
                    mitigation_techniques=['threshold_adjustment'],
                    generate_visualizations=True
                )
                all_results[model['model_name']] = results
            except Exception as e:
                print(f"Error auditing {model['model_name']}: {e}")
                all_results[model['model_name']] = {'error': str(e)}
        
        # Save consolidated report
        consolidated_path = "data/bias_reports/consolidated_audit_report.json"
        os.makedirs(os.path.dirname(consolidated_path), exist_ok=True)
        
        with open(consolidated_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'models_audited': list(all_results.keys()),
                'results': {
                    model: {
                        'status': 'SUCCESS' if 'detection_report' in results else 'ERROR',
                        'report_path': results.get('report_path'),
                        'visualizations_generated': results.get('visualizations_generated', False)
                    }
                    for model, results in all_results.items()
                }
            }, f, indent=2)
        
        print(f"\n\n{'='*80}")
        print("ALL AUDITS COMPLETE")
        print(f"{'='*80}")
        print(f"\nConsolidated report: {consolidated_path}")
        
        return all_results


def main():
    """Run bias audit pipeline with command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Bias Detection and Mitigation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with model selection (RECOMMENDED)
  python bias_pipeline.py
  
  # Run without model selection
  python bias_pipeline.py --no-model-selection
  
  # Run with custom weights (70% performance, 30% fairness)
  python bias_pipeline.py --performance-weight 0.7 --fairness-weight 0.3
  
  # Run without visualizations (faster)
  python bias_pipeline.py --no-visualizations
        """
    )
    
    parser.add_argument(
        '--no-model-selection',
        action='store_true',
        help='Run individual audits without model selection'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip visualization generation (faster execution)'
    )
    
    parser.add_argument(
        '--performance-weight',
        type=float,
        default=0.6,
        help='Weight for performance metrics (default: 0.6)'
    )
    
    parser.add_argument(
        '--fairness-weight',
        type=float,
        default=0.4,
        help='Weight for fairness metrics (default: 0.4)'
    )
    
    parser.add_argument(
        '--min-fairness',
        type=float,
        default=60.0,
        help='Minimum fairness threshold (default: 60.0)'
    )
    
    args = parser.parse_args()
    
    # Validate weights
    if abs(args.performance_weight + args.fairness_weight - 1.0) > 0.01:
        print("Error: Performance weight + Fairness weight must equal 1.0")
        return
    
    print("\n" + "="*80)
    print("BIAS DETECTION AND MITIGATION PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model Selection: {'ENABLED' if not args.no_model_selection else 'DISABLED'}")
    print(f"  Visualizations: {'ENABLED' if not args.no_visualizations else 'DISABLED'}")
    if not args.no_model_selection:
        print(f"  Performance Weight: {args.performance_weight}")
        print(f"  Fairness Weight: {args.fairness_weight}")
        print(f"  Min Fairness Threshold: {args.min_fairness}")
    print()
    
    if not args.no_model_selection:
        # Run with model selection
        pipeline = BiasAuditPipeline()
        
        models_to_audit = [
            {
                'model_name': 'boosted_tree_regressor',
                'predictions_table': f"{pipeline.project_id}.{pipeline.dataset_id}.boosted_tree_rating_predictions"
            },
            {
                'model_name': 'automl_regressor',
                'predictions_table': f"{pipeline.project_id}.{pipeline.dataset_id}.automl_rating_predictions"
            },
            {
                'model_name': 'matrix_factorization',
                'predictions_table': f"{pipeline.project_id}.{pipeline.dataset_id}.matrix_factorization_rating_predictions"
            }
        ]
        
        results = pipeline.run_model_selection(
            model_candidates=models_to_audit,
            performance_weight=args.performance_weight,
            fairness_weight=args.fairness_weight,
            min_fairness_threshold=args.min_fairness,
            generate_visualizations=not args.no_visualizations
        )
        
        print(f"\n\n{'='*80}")
        print("🎉 PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"\n🎯 SELECTED MODEL: {results['selected_model']}")
        print(f"   Production Table: {results['selected_table']}")
        print(f"\n📊 Generated Files:")
        print(f"   - Model Selection Report: data/bias_reports/model_selection_report.json")
        print(f"   - Model Comparison Chart: data/bias_reports/model_selection/model_comparison.png")
        if not args.no_visualizations:
            print(f"   - Fairness Visualizations: data/bias_reports/visualizations/")
        print()
    else:
        # Run without model selection
        run_bias_audit_for_all_models(with_model_selection=False)


if __name__ == "__main__":
    main()
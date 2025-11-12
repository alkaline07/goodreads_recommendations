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
from bias_detection import BiasDetector, BiasReport
from bias_mitigation import BiasMitigator, MitigationResult


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
        self.project_id = self.detector.project_id
        self.dataset_id = "books"
        
        print(f"BiasAuditPipeline initialized for project: {self.project_id}")
    
    def run_full_audit(
        self,
        model_name: str,
        predictions_table: str,
        apply_mitigation: bool = True,
        mitigation_techniques: List[str] = None
    ) -> Dict:
        """
        Run a complete bias audit on a model.
        
        Args:
            model_name: Name of the model
            predictions_table: BigQuery table with predictions
            apply_mitigation: Whether to apply mitigation
            mitigation_techniques: List of techniques to apply
                ['shrinkage', 'threshold_adjustment', 'reweighting']
        
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
            'mitigation_results': [],
            'final_validation': None
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
        
        # Step 2: Apply Mitigation (if requested and bias detected)
        if apply_mitigation and detection_report.disparity_analysis['detailed_disparities']:
            print("\n[STEP 2/4] Applying Bias Mitigation...")
            
            if mitigation_techniques is None:
                mitigation_techniques = ['threshold_adjustment']
            
            for technique in mitigation_techniques:
                if technique == 'threshold_adjustment':
                    mitigation_result = self._apply_threshold_mitigation(
                        predictions_table,
                        model_name,
                        detection_report
                    )
                    audit_results['mitigation_results'].append(mitigation_result)
                    
                elif technique == 'shrinkage':
                    mitigation_result = self._apply_shrinkage_mitigation(
                        predictions_table,
                        model_name,
                        detection_report
                    )
                    audit_results['mitigation_results'].append(mitigation_result)
        else:
            print("\n[STEP 2/4] Skipping mitigation (no significant bias detected or not requested)")
        
        # Step 3: Validate Mitigation
        if audit_results['mitigation_results']:
            print("\n[STEP 3/4] Validating Mitigation Effectiveness...")
            validation = self._validate_mitigation(
                audit_results['mitigation_results'],
                model_name
            )
            audit_results['final_validation'] = validation
        else:
            print("\n[STEP 3/4] Skipping validation (no mitigation applied)")
        
        # Step 4: Generate Comprehensive Report
        print("\n[STEP 4/4] Generating Comprehensive Audit Report...")
        report_path = self._generate_comprehensive_report(audit_results, model_name)
        audit_results['report_path'] = report_path
        
        print("\n" + "="*80)
        print("BIAS AUDIT COMPLETE")
        print("="*80)
        print(f"\nFull audit report: {report_path}")
        
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


def run_bias_audit_for_all_models():
    """Run bias audit for all trained models."""
    pipeline = BiasAuditPipeline()
    
    models_to_audit = [
        {
            'name': 'boosted_tree_regressor',
            'predictions_table': f"{pipeline.project_id}.{pipeline.dataset_id}.boosted_tree_rating_predictions"
        },
        {
            'name': 'automl_regressor',
            'predictions_table': f"{pipeline.project_id}.{pipeline.dataset_id}.automl_rating_predictions"
        }
    ]
    
    all_results = {}
    
    for model in models_to_audit:
        print(f"\n\n{'#'*80}")
        print(f"# AUDITING MODEL: {model['name']}")
        print(f"{'#'*80}\n")
        
        try:
            results = pipeline.run_full_audit(
                model_name=model['name'],
                predictions_table=model['predictions_table'],
                apply_mitigation=True,
                mitigation_techniques=['threshold_adjustment']
            )
            all_results[model['name']] = results
        except Exception as e:
            print(f"Error auditing {model['name']}: {e}")
            all_results[model['name']] = {'error': str(e)}
    
    # Save consolidated report
    consolidated_path = "data/bias_reports/consolidated_audit_report.json"
    os.makedirs(os.path.dirname(consolidated_path), exist_ok=True)
    
    with open(consolidated_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'models_audited': list(all_results.keys()),
            'results': {
                model: {
                    'status': results.get('executive_summary', {}).get('overall_status', 'ERROR'),
                    'report_path': results.get('report_path')
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
    """Run bias audit pipeline."""
    run_bias_audit_for_all_models()


if __name__ == "__main__":
    main()

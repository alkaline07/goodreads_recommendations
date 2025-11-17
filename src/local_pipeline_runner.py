from src.bq_model_training import BigQueryMLModelTraining
from src.register_bqml_models import RegisterBQMLModels
from src.generate_prediction_tables import BiasReadyPredictionGenerator
from src.model_evaluation_pipeline import ModelEvaluationPipeline
from src.bias_pipeline import BiasAuditPipeline
from src.model_validation import BigQueryModelValidator
from src.model_manager import ModelManager
import itertools

if __name__ == "__main__":
    num_factors = [10, 20, 25, 35, 50]
    l2_reg = [0.01, 0.05, 0.1, 0.2, 0.5]
    max_iterations = [20, 30, 40, 50, 60]
    max_tree_depth = [4, 5, 6, 7, 8]
    num_parallel_tree = [10, 25, 50, 75, 100]
    trainer = BigQueryMLModelTraining()
    mf_hyperparams = {
        "num_factors": 25,
        "l2_reg": 0.05,
        "max_iterations": 40
    }
    bt_hyperparams = {
        "max_tree_depth": 6,
        "num_parallel_tree": 10
    }
    mf_param_grid = itertools.product(num_factors, l2_reg, max_iterations)

    for factors, reg, iterations in mf_param_grid:
        current_mf_params = {
            "num_factors": factors,
            "l2_reg": reg,
            "max_iterations": iterations
        }
        
        print(f"Running MF: {current_mf_params}")

        trainer.run(mf_hyperparams=current_mf_params, bt_hyperparams=bt_hyperparams, model_type="matrix_factorization")

        print("Completed MF run")

        registrar = RegisterBQMLModels()
        registrar.register_models()
        predictor = BiasReadyPredictionGenerator()
        predictor.generate_prediction_tables()
        evaluator = ModelEvaluationPipeline()
        evaluator.run_evaluation_pipeline()
        bias_pipeline = BiasAuditPipeline()
        bias_pipeline.run_bias_analysis()
        validator = BigQueryModelValidator(project_id=None)
        ok = validator.validate_selected_model()
        if not (ok):
            print("Validation FAILED — stopping pipeline.")
            exit(1)
        rollback_manager = ModelManager()
        rollback_manager.run_model_rollback_checks()
        print("Pipeline completed successfully.")
        print("-" * 40)
    
    bt_param_grid = itertools.product(max_tree_depth, num_parallel_tree)

    for depth, num_trees in bt_param_grid:
        current_bt_params = {
            "max_tree_depth": depth,
            "num_parallel_tree": num_trees
        }
        
        print(f"Running BT: {current_bt_params}")
    
        trainer.run(mf_hyperparams=mf_hyperparams, bt_hyperparams=current_bt_params, model_type="boosted_tree")

        print("Completed BT run")

        registrar = RegisterBQMLModels()
        registrar.register_models()
        predictor = BiasReadyPredictionGenerator()
        predictor.generate_prediction_tables()
        evaluator = ModelEvaluationPipeline()
        evaluator.run_evaluation_pipeline()
        bias_pipeline = BiasAuditPipeline()
        bias_pipeline.run_bias_analysis()
        validator = BigQueryModelValidator(project_id=None)
        ok = validator.validate_selected_model()
        if not (ok):
            print("Validation FAILED — stopping pipeline.")
            exit(1)
        rollback_manager = ModelManager()
        rollback_manager.run_model_rollback_checks()
        print("Pipeline completed successfully.")
        print("-" * 40)

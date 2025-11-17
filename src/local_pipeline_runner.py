from src.bq_model_training import BigQueryMLModelTraining
from src.register_bqml_models import RegisterBQMLModels
from src.generate_prediction_tables import BiasReadyPredictionGenerator
from src.model_evaluation_pipeline import ModelEvaluationPipeline
from src.bias_pipeline import BiasAuditPipeline
from src.model_validation import BigQueryModelValidator
from src.model_manager import ModelManager
import itertools

if __name__ == "__main__":
    mf_parameter_list = [
        (10, 0.05, 40),
        (20, 0.05, 40),
        (25, 0.05, 40),
        (35, 0.05, 40),
        (40, 0.05, 40),
        (25, 0.01, 40),
        (25, 0.1, 40),
        (25, 0.2, 40),
        (25, 0.5, 40),
        (25, 0.05, 20),
        (25, 0.05, 30),
        (25, 0.05, 50),
        (25, 0.05, 60)
    ]
    mf_hyperparams = {
        "num_factors": 25,
        "l2_reg": 0.05,
        "max_iterations": 40
    }
    bt_hyperparams = {
        "max_tree_depth": 6,
        "num_parallel_tree": 10
    }
    bt_param_list = [
        (4, 10),
        (5, 10),
        (6, 10),
        (7, 10),
        (8, 10),
        (6, 25),
        (6, 50),
        (6, 75),
        (6, 100)
    ]

    for mf_param in mf_parameter_list:
        factors, reg, iterations = mf_param
        print(f"Running MF: num_factors={factors}, l2_reg={reg}, max_iterations={iterations}")
        trainer = BigQueryMLModelTraining()
        trainer.run(mf_hyperparams={
            "num_factors": factors,
            "l2_reg": reg,
            "max_iterations": iterations
        }, bt_hyperparams=bt_hyperparams, model_type="matrix_factorization")
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
    
    for bt_param in bt_param_list:
        depth, num_trees = bt_param
        print(f"Running BT: max_tree_depth={depth}, num_parallel_tree={num_trees}")
        trainer = BigQueryMLModelTraining()
        trainer.run(mf_hyperparams=mf_hyperparams, bt_hyperparams={
            "max_tree_depth": depth,
            "num_parallel_tree": num_trees
        }, model_type="boosted_tree")
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

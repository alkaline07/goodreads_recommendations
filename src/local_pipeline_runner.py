from src.bq_model_training import BigQueryMLModelTraining
from src.register_bqml_models import RegisterBQMLModels
from src.generate_prediction_tables import BiasReadyPredictionGenerator
from src.model_evaluation_pipeline import main as ModelEvaluationPipelineMain
from src.bias_pipeline import main as BiasAuditPipelineMain
from src.model_validation import BigQueryModelValidator
from src.model_manager import main as ModelManagerMain

if __name__ == "__main__":
    mf_parameter_list = [
        (10, 0.05, 40),
        (20, 0.05, 40),
        (25, 0.05, 40),
        (35, 0.05, 40),
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
    ]

    for mf_param in mf_parameter_list:
        factors, reg, iterations = mf_param
        print(f"Running MF: num_factors={factors}, l2_reg={reg}, max_iterations={iterations}")
        trainer = BigQueryMLModelTraining()
        trainer.run(mf_hyperparams={
            "num_factors": factors,
            "l2_reg": reg,
            "max_iterations": iterations
        }, bt_hyperparams=bt_hyperparams, model_type="MATRIX_FACTORIZATION")
        print("Completed MF run")

        registrar = RegisterBQMLModels()
        registrar.main()
        predictor = BiasReadyPredictionGenerator()
        predictor.run(model_types=['boosted_tree', 'matrix_factorization'])
        ModelEvaluationPipelineMain()
        BiasAuditPipelineMain()
        validator = BigQueryModelValidator(project_id=None)
        ok = validator.validate_selected_model()
        if not (ok):
            print("Validation FAILED — stopping pipeline.")
            exit(1)
        ModelManagerMain()
        print("Pipeline completed successfully.")
        print("-" * 40)
    
    for bt_param in bt_param_list:
        depth, num_trees = bt_param
        print(f"Running BT: max_tree_depth={depth}, num_parallel_tree={num_trees}")
        trainer = BigQueryMLModelTraining()
        trainer.run(mf_hyperparams=mf_hyperparams, bt_hyperparams={
            "max_tree_depth": depth,
            "num_parallel_tree": num_trees
        }, model_type="BOOSTED_TREE_REGRESSOR")
        print("Completed BT run")

        registrar = RegisterBQMLModels()
        registrar.main()
        predictor = BiasReadyPredictionGenerator()
        predictor.run(model_types=['boosted_tree', 'matrix_factorization'])
        ModelEvaluationPipelineMain()
        BiasAuditPipelineMain()
        validator = BigQueryModelValidator(project_id=None)
        ok = validator.validate_selected_model()
        if not (ok):
            print("Validation FAILED — stopping pipeline.")
            exit(1)
        ModelManagerMain()
        print("Pipeline completed successfully.")
        print("-" * 40)

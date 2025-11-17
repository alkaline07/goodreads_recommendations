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
        (25, 0.05, 40),
        (40, 0.05, 40),
        (25, 0.01, 40),
        (25, 0.1, 40),
        (25, 0.5, 40),
        (25, 0.05, 20),
        (25, 0.05, 50),
        (25, 0.05, 60)
    ]

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

    for mf_param, bt_param in zip(mf_parameter_list, bt_param_list):
        factors, reg, iterations = mf_param
        depth, n_trees = bt_param
        print(f"Running MF: num_factors={factors}, l2_reg={reg}, max_iterations={iterations}")
        trainer = BigQueryMLModelTraining()
        trainer.run(
            mf_hyperparams={
                "num_factors": factors,
                "l2_reg": reg,
                "max_iterations": iterations
            },
            bt_hyperparams={
                "max_tree_depth": depth,
                "num_parallel_tree": n_trees
            }
        )
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

import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(version_base=None, config_name='config', config_path='.')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]
    
    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Get the absolute path to the project root
    root_path = os.path.dirname(os.path.abspath(__file__))

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(root_path, config['main']['components_repository'], "get_data"),
                "main",
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "basic_cleaning"),
                "main",
                env_manager="conda",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_data",
                    "output_description": "Data after basic cleaning",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, "src", "data_check"),
                "main",
                env_manager="conda",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:latest",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(root_path, config['main']['components_repository'], "train_val_test_split"),
                "main",
                env_manager="conda",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify": config["modeling"]["stratify_by"]
                },
            )

        if "train_random_forest" in active_steps:

            # Path to the train_random_forest project
            train_rf_path = os.path.join(root_path, "src", "train_random_forest")

            # Create rf_config.json *inside* that project directory
            rf_config_path = os.path.join(train_rf_path, "rf_config.json")
            with open(rf_config_path, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # Now run the step, referring to it just as "rf_config.json"
            _ = mlflow.run(
                train_rf_path,
                "main",
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": "rf_config.json",  # now it’s in the CWD of that project
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "trained_random_forest.csv",
                },
            )

        if "test_regression_model" in active_steps:

            _ = mlflow.run(
                os.path.join(root_path, config['main']['components_repository'], "test_regression_model"),
                "main",
                env_manager="conda",
                parameters={
                    "test_dataset": "test_data.csv:latest",
                    "mlflow_model": "trained_random_forest.csv:latest"
                },
            )


if __name__ == "__main__":
    go()

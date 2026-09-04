from pathlib import Path
import json
import os

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
ARTIFACTS_DIR = BASE_DIR / "mlartifacts"
EXPERIMENT_NAME = "CP4_MLP_Wine_Classification"
RANDOM_STATE = 42


def load_dataset():
    """Carrega o dataset Wine e retorna features, target e metadados."""
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target

    dataset_info = {
        "name": "Wine dataset",
        "source": "sklearn.datasets.load_wine",
        "problem_type": "classification",
        "target": "wine_class",
        "rows": int(X.shape[0]),
        "features": list(X.columns),
        "target_names": list(wine.target_names),
        "missing_values_total": int(X.isna().sum().sum()),
    }

    return X, y, dataset_info


def build_pipeline(config):
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=config["hidden_layer_sizes"],
                    activation=config["activation"],
                    solver=config["solver"],
                    learning_rate_init=config["learning_rate_init"],
                    batch_size=config["batch_size"],
                    max_iter=config["max_iter"],
                    alpha=config["alpha"],
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def count_total_neurons(hidden_layer_sizes):
    return sum(hidden_layer_sizes)


def train_and_log_run(config, X_train, X_test, y_train, y_test, dataset_info):
    run_name = config["run_name"]
    pipeline = build_pipeline(config)

    with mlflow.start_run(run_name=run_name) as run:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }

        mlflow.log_param("dataset", dataset_info["name"])
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("hidden_layers", str(config["hidden_layer_sizes"]))
        mlflow.log_param("num_hidden_layers", len(config["hidden_layer_sizes"]))
        mlflow.log_param("total_neurons", count_total_neurons(config["hidden_layer_sizes"]))
        mlflow.log_param("activation", config["activation"])
        mlflow.log_param("solver", config["solver"])
        mlflow.log_param("learning_rate_init", config["learning_rate_init"])
        mlflow.log_param("batch_size", config["batch_size"])
        mlflow.log_param("max_iter", config["max_iter"])
        mlflow.log_param("alpha", config["alpha"])

        mlflow.log_metrics(metrics)

        artifact_dir = BASE_DIR / "work" / "artifacts" / run_name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = artifact_dir / "dataset_info.json"
        metrics_path = artifact_dir / "metrics.json"

        dataset_path.write_text(json.dumps(dataset_info, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        mlflow.log_artifact(str(dataset_path), artifact_path="dataset")
        mlflow.log_artifact(str(metrics_path), artifact_path="results")

        input_example = X_train.head(1)
        signature = infer_signature(X_train, pipeline.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

        result = {
            "run_name": run_name,
            "run_id": run.info.run_id,
            "model_uri": f"runs:/{run.info.run_id}/model",
            "params": {
                "hidden_layers": str(config["hidden_layer_sizes"]),
                "num_hidden_layers": len(config["hidden_layer_sizes"]),
                "total_neurons": count_total_neurons(config["hidden_layer_sizes"]),
                "activation": config["activation"],
                "solver": config["solver"],
                "learning_rate_init": config["learning_rate_init"],
                "batch_size": config["batch_size"],
                "max_iter": config["max_iter"],
                "alpha": config["alpha"],
            },
            "metrics": metrics,
        }

    return result


def main():
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.resolve().as_posix()}"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        client.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=ARTIFACTS_DIR.resolve().as_uri(),
        )

    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y, dataset_info = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    configs = [
        {
            "run_name": "MLP-01",
            "hidden_layer_sizes": (32,),
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.001,
            "batch_size": 16,
            "max_iter": 800,
            "alpha": 0.0001,
        },
        {
            "run_name": "MLP-02",
            "hidden_layer_sizes": (32, 32),
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.001,
            "batch_size": 16,
            "max_iter": 800,
            "alpha": 0.0001,
        },
        {
            "run_name": "MLP-03",
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.001,
            "batch_size": 16,
            "max_iter": 800,
            "alpha": 0.0001,
        },
        {
            "run_name": "MLP-04",
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.0005,
            "batch_size": 16,
            "max_iter": 1000,
            "alpha": 0.0001,
        },
        {
            "run_name": "MLP-05",
            "hidden_layer_sizes": (64, 32),
            "activation": "tanh",
            "solver": "adam",
            "learning_rate_init": 0.001,
            "batch_size": 16,
            "max_iter": 800,
            "alpha": 0.001,
        },
    ]

    results = []

    print(f"Experimento MLflow: {EXPERIMENT_NAME}")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print("Treinando modelos MLP...\n")

    for config in configs:
        result = train_and_log_run(config, X_train, X_test, y_train, y_test, dataset_info)
        results.append(result)
        print(
            f"{result['run_name']} | "
            f"accuracy={result['metrics']['accuracy']:.4f} | "
            f"f1_macro={result['metrics']['f1_macro']:.4f}"
        )

    best_result = max(
        results,
        key=lambda item: (
            item["metrics"]["f1_macro"],
            item["metrics"]["accuracy"],
            -item["params"]["num_hidden_layers"],
            -item["params"]["total_neurons"],
        ),
    )

    best_model_info = {
        "experiment_name": EXPERIMENT_NAME,
        "tracking_uri": mlflow.get_tracking_uri(),
        "selection_metric": "f1_macro",
        "selection_rule": "Maior f1_macro; em caso de empate, maior accuracy e arquitetura mais simples.",
        "best_run_name": best_result["run_name"],
        "best_run_id": best_result["run_id"],
        "model_uri": best_result["model_uri"],
        "dataset": dataset_info,
        "feature_names": dataset_info["features"],
        "target_names": dataset_info["target_names"],
        "params": best_result["params"],
        "metrics": best_result["metrics"],
    }

    best_path = BASE_DIR / "best_model_info.json"
    best_path.write_text(json.dumps(best_model_info, indent=2), encoding="utf-8")

    print("\nMelhor modelo selecionado:")
    print(f"Run: {best_result['run_name']} ({best_result['run_id']})")
    print(f"Model URI: {best_result['model_uri']}")
    print(f"F1 macro: {best_result['metrics']['f1_macro']:.4f}")
    print(f"Accuracy: {best_result['metrics']['accuracy']:.4f}")
    print(f"Informacoes salvas em: {best_path}")


if __name__ == "__main__":
    main()

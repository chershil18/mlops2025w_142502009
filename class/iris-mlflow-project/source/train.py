# src/train.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_and_explore_data():
    """Load and explore the Iris dataset"""
    print("Loading Iris dataset...")
    # Load dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')

    # Create DataFrame for easier handling
    df = X.copy()
    df['species'] = y
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Classes: {iris.target_names}")
    print(f"Class distribution:\n{df['species_name'].value_counts()}")
    return X, y, df


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """Split and preprocess the data"""
    print("Preprocessing data...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    print(f"Training set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def train_random_forest(X_train, y_train, X_test, y_test, run_name_suffix="", **params):
    """Train Random Forest model with MLflow tracking"""
    run_name = f"Random Forest {run_name_suffix}".strip()
    
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)

        # Log training metrics
        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)

        # Log test metrics
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")

        # Create and log confusion matrix plot
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {run_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_filename = f'confusion_matrix_rf_{run.info.run_id[:8]}.png'
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)
        plt.close()
        
        # Clean up temporary file
        if os.path.exists(cm_filename):
            os.remove(cm_filename)

        # Infer model signature and log model
        signature = infer_signature(X_test, y_pred_test)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(3)
        )

        # Log additional tags
        mlflow.set_tag("model_family", "tree_based")
        mlflow.set_tag("dataset", "iris")

        print(f"{run_name} - Test Accuracy: {test_metrics['accuracy']:.4f}")
        return model, test_metrics['accuracy'], run.info.run_id


def train_logistic_regression(X_train, y_train, X_test, y_test, run_name_suffix="", **params):
    """Train Logistic Regression model with MLflow tracking"""
    run_name = f"Logistic Regression {run_name_suffix}".strip()
    
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)

        # Log training metrics
        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)

        # Log test metrics
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        # Create and log confusion matrix plot
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
        plt.title(f'Confusion Matrix - {run_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        cm_filename = f'confusion_matrix_lr_{run.info.run_id[:8]}.png'
        plt.savefig(cm_filename)
        mlflow.log_artifact(cm_filename)
        plt.close()
        
        # Clean up temporary file
        if os.path.exists(cm_filename):
            os.remove(cm_filename)

        # Infer model signature and log model
        signature = infer_signature(X_test, y_pred_test)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(3)
        )

        # Log additional tags
        mlflow.set_tag("model_family", "linear")
        mlflow.set_tag("dataset", "iris")

        print(f"{run_name} - Test Accuracy: {test_metrics['accuracy']:.4f}")
        return model, test_metrics['accuracy'], run.info.run_id


def simple_model_training():
    """Train models without hyperparameter tuning"""
    print("Training individual models...")
    
    # Load and preprocess data
    X, y, df = load_and_explore_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # Set experiment
    mlflow.set_experiment("Iris Classification Simple Training")

    # Train Random Forest
    rf_params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
    rf_model, rf_accuracy, rf_run_id = train_random_forest(
        X_train, y_train, X_test, y_test, **rf_params
    )

    # Train Logistic Regression
    lr_params = {'C': 1.0, 'random_state': 42, 'max_iter': 1000}
    lr_model, lr_accuracy, lr_run_id = train_logistic_regression(
        X_train, y_train, X_test, y_test, **lr_params
    )

    # Determine best model
    if rf_accuracy >= lr_accuracy:
        best_run_id = rf_run_id
        best_model = rf_model
        best_accuracy = rf_accuracy
        print(f"Best model: Random Forest (Accuracy: {best_accuracy:.4f})")
    else:
        best_run_id = lr_run_id
        best_model = lr_model
        best_accuracy = lr_accuracy
        print(f"Best model: Logistic Regression (Accuracy: {best_accuracy:.4f})")

    return best_model, best_run_id, best_accuracy


def hyperparameter_tuning():
    """Perform hyperparameter tuning with nested runs"""
    # Load and preprocess data
    X, y, df = load_and_explore_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # Set experiment
    mlflow.set_experiment("Iris Classification Hyperparameter Tuning")

    best_accuracy = 0
    best_model = None
    best_run_id = None
    best_model_type = None

    print("Starting hyperparameter tuning...")

    # Random Forest hyperparameter combinations
    rf_params_list = [
        {'n_estimators': 50, 'max_depth': 3, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
        {'n_estimators': 200, 'max_depth': 7, 'random_state': 42},
        {'n_estimators': 100, 'max_depth': None, 'random_state': 42}
    ]

    # Train Random Forest variants
    print("Training Random Forest models...")
    for i, params in enumerate(rf_params_list):
        run_name_suffix = f"RF_{i+1}"
        model, accuracy, run_id = train_random_forest(
            X_train, y_train, X_test, y_test, run_name_suffix, **params
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_run_id = run_id
            best_model_type = "Random Forest"

    # Logistic Regression hyperparameter combinations
    lr_params_list = [
        {'C': 0.1, 'random_state': 42, 'max_iter': 1000},
        {'C': 1.0, 'random_state': 42, 'max_iter': 1000},
        {'C': 10.0, 'random_state': 42, 'max_iter': 1000},
    ]

    # Train Logistic Regression variants
    print("Training Logistic Regression models...")
    for i, params in enumerate(lr_params_list):
        run_name_suffix = f"LR_{i+1}"
        model, accuracy, run_id = train_logistic_regression(
            X_train, y_train, X_test, y_test, run_name_suffix, **params
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_run_id = run_id
            best_model_type = "Logistic Regression"

    print(f"\nBest model: {best_model_type} (Accuracy: {best_accuracy:.4f})")
    print(f"Best run ID: {best_run_id}")
    return best_model, best_run_id, best_accuracy


def model_registry_example(best_run_id, model_name="iris-classifier"):
    """Demonstrate model registry functionality"""
    print(f"Registering model from run {best_run_id}...")
    
    try:
        model_uri = f"runs:/{best_run_id}/model"

        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        print(f"Model registered: {model_name}, Version: {model_version.version}")

        # Add model version tags and description
        client = mlflow.tracking.MlflowClient()
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description="Best performing model from training session"
        )

        # Set model version tags
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="stage",
            value="production_candidate"
        )

        print(f"Model {model_name} version {model_version.version} registered successfully")
        return model_name, model_version.version
        
    except Exception as e:
        print(f"Error registering model: {e}")
        print("This might be because the model is already registered. Continuing...")
        return model_name, "unknown"


def load_and_predict(model_name="iris-classifier", stage="Production"):
    """Load model from registry and make predictions"""
    print(f"Loading model {model_name} from {stage} stage...")
    
    try:
        # Load model from registry
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
        print(f"Successfully loaded model from {stage} stage")
    except Exception as e:
        print(f"Error loading model from registry: {e}")
        print("Loading latest version instead...")
        try:
            model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
        except Exception as e2:
            print(f"Error loading latest model: {e2}")
            print("Falling back to training a new model...")
            model, _, _ = simple_model_training()

    # Create sample data for prediction
    X, y, df = load_and_explore_data()
    X_sample = X.head(5)

    # Make predictions
    predictions = model.predict(X_sample)
    prediction_proba = model.predict_proba(X_sample) if hasattr(model, 'predict_proba') else None

    # Display results
    results = pd.DataFrame({
        'sepal_length': X_sample['sepal length (cm)'],
        'sepal_width': X_sample['sepal width (cm)'],
        'petal_length': X_sample['petal length (cm)'],
        'petal_width': X_sample['petal width (cm)'],
        'predicted_class': predictions,
        'predicted_species': [['setosa', 'versicolor', 'virginica'][p] for p in predictions]
    })
    
    if prediction_proba is not None:
        for i, class_name in enumerate(['setosa', 'versicolor', 'virginica']):
            results[f'prob_{class_name}'] = prediction_proba[:, i]
    
    print("\nPrediction Results:")
    print(results.to_string(index=False, float_format='%.3f'))
    return results, model


def main():
    """Main function to run the complete pipeline"""
    print("=== MLflow Iris Classification Pipeline ===\n")

    # Set tracking URI (if using remote server)
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Ensure MLflow tracking is started
    if not mlflow.active_run():
        mlflow.start_run(run_name="Main Pipeline")
        main_run_active = True
    else:
        main_run_active = False

    try:
        # Option 1: Run hyperparameter tuning
        print("1. Running hyperparameter tuning...")
        best_model, best_run_id, best_accuracy = hyperparameter_tuning()
        
        # Option 2: Or run simple training (comment out the above and uncomment below)
        # print("1. Running simple model training...")
        # best_model, best_run_id, best_accuracy = simple_model_training()

        # Register the best model
        print("\n2. Registering best model...")
        model_name, version = model_registry_example(best_run_id)

        # Load and make predictions
        print("\n3. Making predictions with the model...")
        results, loaded_model = load_and_predict(model_name, "Production")

        print(f"\n=== Pipeline Completed Successfully ===")
        print(f"Best model accuracy: {best_accuracy:.4f}")
        print(f"Model registered as: {model_name} (Version: {version})")
        print(f"MLflow UI: http://127.0.0.1:5000")
        
    finally:
        # End the main run if we started it
        if main_run_active:
            mlflow.end_run()


if __name__ == "__main__":
    main()
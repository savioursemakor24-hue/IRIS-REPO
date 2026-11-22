import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

def run():
    print("=" * 60)
    print("IRIS DATASET CLASSIFICATION: TRAINING & EVALUATION")
    print("=" * 60)

    # 1. Load the Iris dataset using scikit-learn
    iris = load_iris(as_frame=True)
    df = iris.frame
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = list(iris.target_names)

    print(f"\n[+] Loaded dataset from scikit-learn:")
    print(f"    - Samples: {len(df)}")
    print(f"    - Features: {feature_names}")
    print(f"    - Target classes: {target_names}")
    print(f"    - Class distribution: {dict(zip(target_names, np.bincount(y)))}")

    # Summary statistics
    summary_stats = df.describe().to_dict()

    # 2. Train-Test Split (utilizing all available 4 features, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[+] Train/Test Split (80/20 stratified):")
    print(f"    - Training samples: {X_train.shape[0]}")
    print(f"    - Testing samples:  {X_test.shape[0]}")

    # 3. Model Candidates
    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Support Vector Machine (SVC)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, kernel="rbf", C=1.0, random_state=42))
        ]),
        "k-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ])
    }

    # 4. Cross-Validation (5-Fold Stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    print("\n[+] 5-Fold Stratified Cross-Validation Benchmark:")
    for name, model in candidates.items():
        scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        acc_mean = scores['test_accuracy'].mean()
        acc_std = scores['test_accuracy'].std()
        f1_mean = scores['test_f1_macro'].mean()
        cv_results[name] = {
            "accuracy_mean": float(acc_mean),
            "accuracy_std": float(acc_std),
            "f1_macro_mean": float(f1_mean),
            "scores": [float(s) for s in scores['test_accuracy']]
        }
        print(f"    - {name:28s}: Accuracy = {acc_mean:.4f} (+/- {acc_std:.4f}), F1-Macro = {f1_mean:.4f}")

    # 5. Select Best Model & Train on Training Set
    # We choose Random Forest as our primary production classifier for its high accuracy, robustness, and interpretability
    chosen_name = "Random Forest"
    best_model = candidates[chosen_name]
    best_model.fit(X_train, y_train)

    # Also fit Logistic Regression for probability coefficients comparison
    lr_model = candidates["Logistic Regression"]
    lr_model.fit(X_train, y_train)

    # 6. Evaluation on Test Set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision_macro = precision_score(y_test, y_pred, average="macro")
    test_precision_weighted = precision_score(y_test, y_pred, average="weighted")
    test_recall_macro = recall_score(y_test, y_pred, average="macro")
    test_recall_weighted = recall_score(y_test, y_pred, average="weighted")
    test_f1_macro = f1_score(y_test, y_pred, average="macro")
    test_f1_weighted = f1_score(y_test, y_pred, average="weighted")
    test_roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

    cm = confusion_matrix(y_test, y_pred).tolist()
    cls_report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS FOR {chosen_name.upper()} (TEST SET)")
    print("=" * 60)
    print(f"Test Accuracy:          {test_accuracy * 100:.2f}%")
    print(f"Precision (Macro):      {test_precision_macro:.4f}")
    print(f"Precision (Weighted):   {test_precision_weighted:.4f}")
    print(f"Recall (Macro):         {test_recall_macro:.4f}")
    print(f"Recall (Weighted):      {test_recall_weighted:.4f}")
    print(f"F1-Score (Macro):       {test_f1_macro:.4f}")
    print(f"F1-Score (Weighted):    {test_f1_weighted:.4f}")
    print(f"ROC-AUC (One-vs-Rest):  {test_roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(cm))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Feature Importance (Random Forest)
    importances = best_model.feature_importances_.tolist()
    feature_importance_dict = dict(zip(feature_names, [round(val, 4) for val in importances]))
    print("Feature Importances:")
    for feat, imp in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {feat:20s}: {imp:.4f} ({imp*100:.1f}%)")

    # 7. Model Persistence
    output_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "iris_rf_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\n[+] Saved serialized model artifact to: {model_path}")

    # Retrain on full dataset for final deployment artifact
    full_rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    full_rf.fit(X, y)
    full_model_path = os.path.join(models_dir, "iris_rf_model_full.joblib")
    joblib.dump(full_rf, full_model_path)
    print(f"[+] Saved full production model artifact to: {full_model_path}")

    # Also save Logistic Regression full pipeline
    full_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    full_lr.fit(X, y)
    joblib.dump(full_lr, os.path.join(models_dir, "iris_lr_model_full.joblib"))

    # 8. Model Registry and Metadata Export
    metadata = {
        "model_name": "Iris Multi-Class Classifier",
        "model_version": "1.0.0",
        "algorithm": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 4,
            "random_state": 42
        },
        "dataset": {
            "name": "Fisher's Iris Dataset (via scikit-learn)",
            "total_samples": len(df),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "feature_ranges": {
                "sepal length (cm)": {"min": float(df["sepal length (cm)"].min()), "max": float(df["sepal length (cm)"].max()), "mean": float(df["sepal length (cm)"].mean())},
                "sepal width (cm)": {"min": float(df["sepal width (cm)"].min()), "max": float(df["sepal width (cm)"].max()), "mean": float(df["sepal width (cm)"].mean())},
                "petal length (cm)": {"min": float(df["petal length (cm)"].min()), "max": float(df["petal length (cm)"].max()), "mean": float(df["petal length (cm)"].mean())},
                "petal width (cm)": {"min": float(df["petal width (cm)"].min()), "max": float(df["petal width (cm)"].max()), "mean": float(df["petal width (cm)"].mean())}
            },
            "target_names": target_names,
            "target_mapping": {0: "setosa", 1: "versicolor", 2: "virginica"}
        },
        "evaluation_metrics": {
            "test_accuracy": float(test_accuracy),
            "precision_macro": float(test_precision_macro),
            "precision_weighted": float(test_precision_weighted),
            "recall_macro": float(test_recall_macro),
            "recall_weighted": float(test_recall_weighted),
            "f1_macro": float(test_f1_macro),
            "f1_weighted": float(test_f1_weighted),
            "roc_auc_ovr": float(test_roc_auc),
            "cross_validation_5fold": cv_results
        },
        "confusion_matrix": {
            "labels": target_names,
            "matrix": cm
        },
        "classification_report": cls_report,
        "feature_importances": feature_importance_dict,
        "artifacts": {
            "model_joblib": "models/iris_rf_model_full.joblib",
            "eval_model_joblib": "models/iris_rf_model.joblib"
        }
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved model registry metadata to: {metadata_path}")

    # Export a lightweight portable client model weights/estimators representation
    # for client-side or micro-serverless scoring without cold start
    # We can also extract decision trees thresholds or logistic regression weights:
    lr_clf = full_lr.named_steps["clf"]
    scaler = full_lr.named_steps["scaler"]
    portable_model = {
        "classes": target_names,
        "features": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "lr_coef": lr_clf.coef_.tolist(),
        "lr_intercept": lr_clf.intercept_.tolist(),
        "feature_importances": feature_importance_dict
    }
    with open(os.path.join(output_dir, "portable_model.json"), "w") as f:
        json.dump(portable_model, f, indent=2)
    print(f"[+] Saved portable model definition to: {os.path.join(output_dir, 'portable_model.json')}")

    print("\n[SUCCESS] Model training, evaluation, and serialization completed successfully!")

if __name__ == "__main__":
    run()

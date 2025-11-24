import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

from app.ml.data_loader import load_csv
from app.ml.preprocess import build_preprocessing_pipeline, get_numeric_feature_names

MODEL_FILE = "fraud_model_pipeline.pkl"
META_FILE = "fraud_model_meta.pkl"

def train_and_save_model(csv_path: str, model_dir: str = "models", target_col: str = "Class", test_size: float = 0.2, random_state: int = 42):
    df = load_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    # Separate X, y
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # Numeric features
    numeric_features = get_numeric_feature_names(X, exclude=[])

    # Preprocessor
    preprocessor = build_preprocessing_pipeline(numeric_features)

    # Train-test split (stratify due to imbalance)
    X_train, X_test, y_train, y_test = train_test_split(X[numeric_features], y, test_size=test_size, stratify=y, random_state=random_state)

    # Handle imbalance with SMOTE applied on training set only
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Classifier 
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1, class_weight="balanced")

    # Full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    # Fit on resampled training data
    pipeline.fit(X_res, y_res)

    # Predict on test
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # Metrics
    auc = float(roc_auc_score(y_test, y_proba))
    f1 = float(f1_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))

    # Save model and metadata
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, MODEL_FILE)
    meta_path = os.path.join(model_dir, META_FILE)

    joblib.dump(pipeline, model_path)
    metadata = {
        "numeric_features": numeric_features,
        "target_col": target_col
    }
    joblib.dump(metadata, meta_path)

    return {
        "train_samples": int(len(X_res)),
        "test_samples": int(len(X_test)),
        "auc": auc,
        "f1": f1,
        "model_path": model_path
    }

def evaluate_saved_model(csv_path: str, model_dir: str = "models"):
    model_path = os.path.join(model_dir, MODEL_FILE)
    meta_path = os.path.join(model_dir, META_FILE)
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Model or metadata not found. Train first.")

    pipeline = joblib.load(model_path)
    metadata = joblib.load(meta_path)
    target_col = metadata["target_col"]
    numeric_features = metadata["numeric_features"]

    df = load_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df[numeric_features]
    y = df[target_col].astype(int)

    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)

    auc = float(roc_auc_score(y, y_proba))
    f1 = float(f1_score(y, y_pred))
    precision = float(precision_score(y, y_pred))
    recall = float(recall_score(y, y_pred))

    return {
        "samples": int(len(X)),
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

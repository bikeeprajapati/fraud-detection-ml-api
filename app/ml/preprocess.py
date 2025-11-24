from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Tuple, List
import pandas as pd
import numpy as np

def build_preprocessing_pipeline(numeric_features: List[str]):
    """
    Returns ColumnTransformer or Pipeline for numeric features.
    For fraud detection (mostly numeric), we impute and scale.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features)
    ], remainder="drop")
    return preprocessor

def get_numeric_feature_names(df: pd.DataFrame, exclude: List[str]):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude]

import os
import joblib
import pandas as pd
from typing import List, Dict

MODEL_FILE = "fraud_model_pipeline.pkl"
META_FILE = "fraud_model_meta.pkl"

def load_model_and_metadata(model_dir: str = "models"):
    model_path = os.path.join(model_dir, MODEL_FILE)
    meta_path = os.path.join(model_dir, META_FILE)
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None
    pipeline = joblib.load(model_path)
    metadata = joblib.load(meta_path)
    return {"pipeline": pipeline, "metadata": metadata}

def predict_from_dicts(records: List[Dict[str, float]], model_meta: dict) -> Dict[str, List]:
    pipeline = model_meta["pipeline"]
    metadata = model_meta["metadata"]
    numeric_features = metadata["numeric_features"]

    # Build DataFrame with required numeric columns (fill missing with 0)
    df = pd.DataFrame(records)
    df = df.reindex(columns=numeric_features, fill_value=0)

    probs = pipeline.predict_proba(df)[:, 1].tolist()
    preds = pipeline.predict(df).astype(int).tolist()

    return {"probabilities": probs, "predictions": preds}

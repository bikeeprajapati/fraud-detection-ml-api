from pydantic import BaseModel
from typing import Dict, List, Optional

class TrainResponse(BaseModel):
    train_samples: int
    test_samples: int
    auc: float
    f1: float
    model_path: str

class EvalResponse(BaseModel):
    samples: int
    auc: float
    f1: float
    precision: float
    recall: float

class PredictRequest(BaseModel):
    record: Optional[Dict[str, float]] = None
    records: Optional[List[Dict[str, float]]] = None

class PredictResponse(BaseModel):
    probabilities: List[float]
    predictions: List[int]

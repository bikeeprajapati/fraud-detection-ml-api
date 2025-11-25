import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import TrainResponse, EvalResponse, PredictRequest, PredictResponse
from app.ml.trainer import train_and_save_model, evaluate_saved_model
from app.ml.predictor import load_model_and_metadata, predict_from_dicts

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = FastAPI(title="Fraud Detection API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# GLOBAL MODEL VARIABLE
model_meta = None


# ----------------------
# Load model at startup
# ----------------------
@app.on_event("startup")
def load_model_startup():
    global model_meta
    try:
        model_meta = load_model_and_metadata(MODEL_DIR)
        print("✅ Model loaded into memory at startup.")
    except Exception:
        print("⚠ No trained model found. Train one with /train.")


# Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# -------------------------
# TRAIN MODEL
# -------------------------
@app.post("/train", response_model=TrainResponse)
def train_endpoint(
    csv_path: str = Query("data/creditcard.csv", description="Path to CSV"),
    target_col: str = Query("Class", description="Target column name")
):
    global model_meta

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail=f"CSV file not found at '{csv_path}'")

    try:
        metrics = train_and_save_model(csv_path, model_dir=MODEL_DIR, target_col=target_col)
        
        # Reload the trained model
        model_meta = load_model_and_metadata(MODEL_DIR)

        return TrainResponse(**metrics)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# UPLOAD DATASET CSV
# -------------------------
@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), filename: str = "creditcard.csv"):
    save_path = os.path.join("data", filename)
    contents = await file.read()
    
    with open(save_path, "wb") as f:
        f.write(contents)
    
    return {"detail": f"Saved dataset to {save_path}"}


# -------------------------
# EVALUATE SAVED MODEL
# -------------------------
@app.get("/evaluate", response_model=EvalResponse)
def evaluate_endpoint(csv_path: str = "data/creditcard.csv"):
    try:
        metrics = evaluate_saved_model(csv_path, model_dir=MODEL_DIR)
        return EvalResponse(**metrics)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=400, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# PREDICT ENDPOINT
# -------------------------
@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    global model_meta

    if model_meta is None:
        raise HTTPException(status_code=400, detail="No trained model found. Train with /train.")

    # Validate input
    if payload.record is None and payload.records is None:
        raise HTTPException(status_code=422, detail="Provide 'record' or 'records' in body.")

    try:
        if payload.record is not None:
            records = [payload.record]
        else:
            records = payload.records

        result = predict_from_dicts(
            records,
            {
                "pipeline": model_meta["pipeline"],
                "metadata": model_meta["metadata"]
            }
        )

        return PredictResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

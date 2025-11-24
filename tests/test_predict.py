import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

# A minimal test that checks predict returns 400 if no model
def test_predict_no_model():
    payload = {"record": {"feature_x": 1.0}}
    r = client.post("/predict", json=payload)
    assert r.status_code in (400, 500)

# 💳 Credit Card Fraud Detection

A Machine Learning project to detect fraudulent credit card transactions using **FastAPI** and **Streamlit**.

---

## Features
- Predict single transactions or batch via CSV
- Interactive Streamlit frontend
- Fraud distribution charts
- Docker-ready deployment

---

## Setup

1. Clone repository:
```bash
git clone https://github.com/bikeeprajapati/fraud-detection-ml-api.git
cd fraud-detection-ml-api
Create virtual environment and install dependencies:

bash
Copy code
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
Run FastAPI:

bash
Copy code
uvicorn app.main:app --reload
API docs: http://127.0.0.1:8000/docs

Run Streamlit:

bash
Copy code
streamlit run streamlit_app/streamlit_app.py
Docker
bash
Copy code
docker build -t fraud-detection-app .
docker run -p 8501:8501 fraud-detection-app
Dataset
Kaggle Credit Card Fraud Detection


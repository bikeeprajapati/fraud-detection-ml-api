FROM python:3.10-slim

WORKDIR /app

# Copy project
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose required ports
EXPOSE 8000
EXPOSE 8501

# Run FastAPI and Streamlit together
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run streamlit_app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0 & \
    wait

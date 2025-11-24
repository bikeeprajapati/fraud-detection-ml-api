import streamlit as st
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("💳 Credit Card Fraud Detection")

# -------------------------------------------
# Single Prediction
# -------------------------------------------
st.header("Single Transaction Prediction")

# Generate blank input fields based on features
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

inputs = {}

cols = st.columns(3)
for i, feat in enumerate(FEATURES):
    inputs[feat] = cols[i % 3].number_input(feat, value=0.0)

if st.button("Predict Single Record"):
    payload = {"record": inputs}

    try:
        resp = requests.post(API_URL, json=payload)
        if resp.status_code == 200:
            result = resp.json()
            st.success("Prediction Successful")
            st.json(result)
        else:
            st.error(f"API Error: {resp.status_code}")
            st.code(resp.text)
    except Exception as e:
        st.error("Could not reach API")
        st.exception(e)

st.markdown("---")

# -------------------------------------------
# Batch Prediction (CSV)
# -------------------------------------------
st.header("Batch Prediction (CSV Upload)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())

        if st.button("Predict Batch"):
            payload = {"records": df.to_dict(orient="records")}

            try:
                resp = requests.post(API_URL, json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success("Batch Prediction Successful")

                    df["probabilities"] = result["probabilities"]
                    df["predictions"] = result["predictions"]

                    st.dataframe(df.head())

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Results CSV", csv, "predictions.csv")
                else:
                    st.error(f"API Error: {resp.status_code}")
                    st.code(resp.text)
            except Exception as e:
                st.error("Could not reach API")
                st.exception(e)

    except Exception as e:
        st.error("Invalid CSV Format")
        st.exception(e)

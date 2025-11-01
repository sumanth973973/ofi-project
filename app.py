import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

st.set_page_config(page_title="📊 OFI Case Study Dashboard", layout="wide")

st.title("📦 OFI Case Study — AI-Driven Logistics Dashboard")
st.caption("Data integration • Predictive analytics • Visualization")

# ------------------------------------------------------------------
# 🔹 Load all data safely
# ------------------------------------------------------------------
@st.cache_data
def load_all_data():
    data = {}
    files = [
        "orders.csv",
        "delivery_performance.csv",
        "routes_distance.csv",
        "cost_breakdown.csv",
        "customer_feedback.csv",
        "warehouse_inventory.csv",
    ]
    for f in files:
        path = f"data/{f}"
        if os.path.exists(path):
            data[f.replace(".csv", "")] = pd.read_csv(path)
        else:
            st.warning(f"⚠️ Missing file: {path}")
    return data

with st.spinner("📂 Loading all datasets..."):
    data = load_all_data()
st.success("✅ All available CSVs loaded!")

# ------------------------------------------------------------------
# 🔹 Show overview of all CSVs
# ------------------------------------------------------------------
st.subheader("📁 Dataset Overview")
for name, df in data.items():
    with st.expander(f"🔸 {name} ({df.shape[0]} rows, {df.shape[1]} columns)"):
        st.dataframe(df.head())

# ------------------------------------------------------------------
# 🔹 Build predictive delivery model (needs 3 files)
# ------------------------------------------------------------------
if all(k in data for k in ["orders", "delivery_performance", "routes_distance"]):
    st.subheader("🚚 Predictive Delivery Delay Model")

    df = data["orders"].merge(data["delivery_performance"], on="order_id", how="left")
    df = df.merge(data["routes_distance"], on="order_id", how="left")

    # check required columns
    if not all(col in df.columns for col in ["actual_delivery_time", "promised_delivery_time"]):
        st.error("❌ Missing time columns in your data. Please check 'orders.csv' and 'delivery_performance.csv'.")
        st.stop()

    df = df.dropna(subset=["actual_delivery_time", "promised_delivery_time"])
    df["delay_minutes"] = (
        pd.to_datetime(df["actual_delivery_time"]) -
        pd.to_datetime(df["promised_delivery_time"])
    ).dt.total_seconds() / 60
    df["is_delayed"] = (df["delay_minutes"] > 0).astype(int)

    features = ["distance_km", "fuel_consumed", "traffic_delay_minutes"]
    if not all(f in df.columns for f in features):
        st.warning("⚠️ Missing route columns — cannot train model.")
    else:
        X = df[features]
        y = df["is_delayed"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        with st.spinner("🤖 Training Random Forest model..."):
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
        st.success(f"✅ Model trained! Accuracy: {acc:.2f}")

        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/delay_model.pkl")

        # Prediction inputs
        st.subheader("🔮 Predict New Delivery Delay")
        distance = st.number_input("Distance (km)", 0.0, 2000.0, 100.0)
        fuel = st.number_input("Fuel Consumed (litres)", 0.0, 200.0, 10.0)
        traffic = st.number_input("Traffic Delay (minutes)", 0.0, 300.0, 15.0)

        if st.button("Predict Delay"):
            model = joblib.load("model/delay_model.pkl")
            pred = model.predict(np.array([[distance, fuel, traffic]]))[0]
            st.error("🚚 Likely DELAYED") if pred else st.success("✅ Likely ON TIME")

        # Charts
        st.subheader("📊 Delivery Delay Analysis")
        fig = px.histogram(df, x="delay_minutes", nbins=30, title="Distribution of Delivery Delays (mins)")
        st.plotly_chart(fig, use_container_width=True)
        if "priority_level" in df.columns:
            fig2 = px.box(df, x="priority_level", y="delay_minutes", title="Delays by Priority Level")
            st.plotly_chart(fig2, use_container_width=True)

else:
    st.error("❌ Missing key datasets for model training — need orders.csv, delivery_performance.csv, and routes_distance.csv.")

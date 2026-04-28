# ---------------------------------
# House Price Prediction Web App
# ---------------------------------

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction System")
st.markdown("Predict house prices using Machine Learning")

# -------------------------------
# CREATE DATA
# -------------------------------

@st.cache_data
def create_data():
    np.random.seed(42)
    n = 500

    data = pd.DataFrame({
        "area": np.random.randint(500, 5000, n),
        "bedrooms": np.random.randint(1, 6, n),
        "bathrooms": np.random.randint(1, 4, n),
        "age": np.random.randint(0, 30, n),
        "parking": np.random.randint(0, 2, n),
        "location": np.random.choice(["urban", "suburban", "rural"], n)
    })

    data["price"] = (
        data["area"] * 300 +
        data["bedrooms"] * 50000 +
        data["bathrooms"] * 30000 -
        data["age"] * 10000 +
        data["parking"] * 20000 +
        np.random.randint(-50000, 50000, n)
    )

    return data

data = create_data()

# -------------------------------
# PREPROCESS
# -------------------------------

data_encoded = pd.get_dummies(data, columns=["location"], drop_first=True)

X = data_encoded.drop("price", axis=1)
y = data_encoded["price"]

# -------------------------------
# TRAIN MODEL
# -------------------------------

@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

model = train_model()

# -------------------------------
# USER INPUT
# -------------------------------

st.header("Enter Property Details")

area = st.slider("Area (sq ft)", 500, 5000, 2000)
bedrooms = st.slider("Bedrooms", 1, 5, 3)
bathrooms = st.slider("Bathrooms", 1, 3, 2)
age = st.slider("Age of House", 0, 30, 5)
parking = st.selectbox("Parking", ["No", "Yes"])
location = st.selectbox("Location", ["urban", "suburban", "rural"])

parking_val = 1 if parking == "Yes" else 0
urban = 1 if location == "urban" else 0
suburban = 1 if location == "suburban" else 0

# -------------------------------
# PREDICT
# -------------------------------

if st.button("Predict Price 💰"):

    features = np.array([[area, bedrooms, bathrooms, age, parking_val, urban, suburban]])
    prediction = model.predict(features)

    st.success(f"🏡 Estimated Price: ₹ {prediction[0]:,.2f}")

    # -------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------
    st.subheader("📊 Feature Importance")

    importances = model.feature_importances_
    feature_names = X.columns

    fig, ax = plt.subplots()
    ax.barh(feature_names, importances)
    ax.set_title("Feature Importance")

    st.pyplot(fig)

# -------------------------------
# OPTIONAL DATA VIEW
# -------------------------------

if st.checkbox("Show Dataset"):
    st.write(data.head())
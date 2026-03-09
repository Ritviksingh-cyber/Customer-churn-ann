import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------- Load Model and Preprocessors ----------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model.h5")

    with open("label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open("onehot_encoder_geo.pkl", "rb") as f:
        onehot_encoder_geo = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()


# ---------- Page Title ----------
st.title("🏦 Customer Churn Prediction System")
st.markdown(
    "Predict whether a **bank customer is likely to churn** using a trained **Artificial Neural Network (ANN)**."
)

st.divider()


# ---------- Sidebar Inputs ----------
st.sidebar.header("📋 Customer Information")

geography = st.sidebar.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.sidebar.slider("Age", 18, 92, 30)
tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5)
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)

balance = st.sidebar.number_input("Balance", min_value=0.0)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=900, value=600)
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0)

has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active_member = st.sidebar.selectbox("Is Active Member", [0, 1])


# ---------- Prepare Input ----------
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})


# ---------- One Hot Encode Geography ----------
geo_encoded = onehot_encoder_geo.transform([[geography]])

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


# ---------- Scale ----------
input_scaled = scaler.transform(input_data)


# ---------- Prediction ----------
if st.button("🔍 Predict Churn"):

    with st.spinner("Running prediction..."):
        prediction = model.predict(input_scaled)

    churn_probability = prediction[0][0]

    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{churn_probability:.2%}")

        st.progress(float(churn_probability))

    with col2:
        if churn_probability > 0.5:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is not likely to churn")


st.divider()

st.markdown(
"""
### 🧠 Model Information
- Model: Artificial Neural Network  
- Framework: TensorFlow / Keras  
- Dataset: Bank Customer Churn Dataset  

This system predicts customer churn probability based on financial and behavioral attributes.
"""
)
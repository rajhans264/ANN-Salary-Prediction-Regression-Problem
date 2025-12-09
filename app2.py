import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os
# -----------------------------
# CACHED LOADERS
# -----------------------------

@st.cache_resource
def load_model_safely():
    # Prefer modern format
    if os.path.exists("SalaryPredictionModel.keras"):
        try:
            model = tf.keras.models.load_model("SalaryPredictionModel.keras")
            return model
        except Exception as e:
            st.warning(f"Failed to load model.keras: {e}")

    # Legacy fallback
    if os.path.exists("SalaryPredictionModel.h5"):
        try:
            model = tf.keras.models.load_model("SalaryPredictionModel.h5", compile=False)
            model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
            return model
        except Exception as e:
            st.error(f"Failed to load SalaryPredictionModel.h5: {e}")

    raise FileNotFoundError("No model file found. Please include SalaryPredictionModel.keras or SalaryPredictionModel.h5 in the repository.")

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# -----------------------------
# LOAD ARTIFACTS
# -----------------------------

try:
    model = load_model_safely()
except Exception as e:
    st.stop()

try:
    label_encoder_gender = load_pickle("LabelEncoderGender.pkl")
    onehot_encoder_geo = load_pickle("OneHotEncoderGeography.pkl")
    scaler = load_pickle("StandardScaler.pkl")
except FileNotFoundError as e:
    st.error(f"Missing artifact: {e}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

# -----------------------------
# -----------------------------

# Streamlit app

st.title('Customer Salary Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92 ,35)
balance = st.number_input('Balance', min_value=0.0, value=0.0, step=100.0)
credit_score = st.number_input('Credit Score', min_value=0, value=600, step=10)
Exited = st.selectbox('Exited', [0, 1], index=1)
tenure = st.slider('Tenure', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1], index=1)
is_active_member = st.selectbox('Is Active Member', [0, 1], index=1)

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [Exited]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
# prediction_proba = prediction[0][0]

# Display the result
st.subheader('Salary Prediction')
st.write(f'Predicted Salary: {prediction[0][0]:.2f}')
# if prediction_proba > 0.5:
#     st.write('The customer is likely to churn.')
# else:
#     st.write('The customer is not likely to churn.')


# to run the app, use the command: streamlit run app2.py
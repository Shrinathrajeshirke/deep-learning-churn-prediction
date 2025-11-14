import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load the trained model
model = tf.keras.models.load_model('model.keras',compile=False)

## Load the encoders and scaler
with open('label_encoder.pkl','rb') as file:
    label_encoder = pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    onehot_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit app
st.title("Customer Churn Prediction")

## User inputs
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of products', 1, 4)
has_cr_card = st.selectbox('Has credit card', [0, 1])
is_active_member = st.selectbox('Is active member', [0, 1])

## Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

## OneHot encode geography
geo_encoded = onehot_encoder.transform([[geography]])
if hasattr(geo_encoded, 'toarray'):
    geo_encoded = geo_encoded.toarray()

geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

## Combine the data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Scale the input data
input_data_scaled = scaler.transform(input_data)

## Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

## Display results
st.subheader("Prediction Results")
st.write(f'**Churn Probability:** {prediction_proba:.2%}')

if prediction_proba > 0.5:
    st.error('The customer is likely to churn')
else:
    st.success('The customer is not likely to churn')

## Optional: Add a probability bar
st.progress(float(prediction_proba))
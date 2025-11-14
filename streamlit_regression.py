import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_reg.keras', compile=False)
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])
    return model

## Load the encoders and scaler
@st.cache_resource
def load_preprocessing_tools():
    with open('encoder.pkl','rb') as file:
        label_encoder = pickle.load(file)
    
    with open('onehotencoder.pkl','rb') as file:
        onehot_encoder = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return label_encoder, onehot_encoder, scaler

model = load_model()
label_encoder, onehot_encoder, scaler = load_preprocessing_tools()

## Streamlit app
st.title("Customer Salary Prediction")

## User inputs
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92, 40)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
tenure = st.slider('Tenure', 0, 10, 3)
num_of_products = st.slider('Number of products', 1, 4, 1)
has_cr_card = st.selectbox('Has credit card', [0, 1])
is_active_member = st.selectbox('Is active member', [0, 1])
exited = st.selectbox('Has Exited', [0, 1])

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
    'Exited': [exited]
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

## Predict salary
prediction = model.predict(input_data_scaled, verbose=0)
predicted_salary = prediction[0][0]

## Display results
st.subheader("Prediction Results")
st.write(f'Predicted Estimated Salary: ${predicted_salary:,.2f}')

if predicted_salary < 75000:
    st.info('Lower salary range')
elif predicted_salary < 125000:
    st.success('Mid salary range')
else:
    st.success('Upper salary range')
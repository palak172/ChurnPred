# Gender -> 1 FEMALE 0 MALE
# Churn -> 1 YES 0 N0
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl 
# order of the columns 'Age', 'Gender', 'Tenure', 'MonthlyCharges']

import streamlit as st
import joblib 
import numpy as np

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

st.title('Churn Prediction App')

st.divider()

st.write('Please enter the values and hit the predict button for getting a prediction.')

age= st.number_input('Enter age', min_value=10, max_value=100, value=30)

tenure = st.number_input('Enter Tenure', min_value=0, max_value=130, value=10)

monthlycharges= st.number_input('Enter Monthly Charge', min_value=30, max_value=150)

gender = st.selectbox("Gender", ["Male", "Female"])

if gender == "Male":
    gender = 0
else:
    gender = 1

st.divider()

predictbutton= st.button('Predict')

if predictbutton:
    X=[age, gender,tenure, monthlycharges]
    X1 = np.array(X)
    X_array= scaler.transform([X1])

    prediction = model.predict(X_array)
    predicted= 'Churn' if prediction == 1 else 'Not churn'
    st.write(f'Predicted: {predicted}')

else:
    st.write('Please enter the values and use Predict button')



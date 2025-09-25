import streamlit as st
import pandas as pd
from src.model import load_model

model = load_model('models/credit_model.pkl')

st.title('Credit Scoring Model')
age = st.number_input('Age', min_value=18, max_value=100)
income = st.number_input('Income')
credit_history = st.selectbox('Credit History', ['Good', 'Bad'])
employment_status = st.selectbox('Employment Status', ['Employed', 'Unemployed'])

if st.button('Predict'):
    # Prepare input data
    input_data = pd.DataFrame([[age, income, credit_history, employment_status]],
                              columns=['Age', 'Income', 'Credit_History', 'Employment_Status'])
    input_data = pd.get_dummies(input_data, drop_first=True)
    prediction = model.predict(input_data)
    st.write(f'Predicted Credit Score: {prediction[0]}')

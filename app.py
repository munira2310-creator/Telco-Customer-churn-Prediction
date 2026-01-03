import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model and features
model = joblib.load('churn_model_final.pkl')
features = joblib.load('model_columns.pkl')

# 2. App Title and Description
st.set_page_config(page_title="Churn Predictor", page_icon="📊")
st.title("Customer Churn Predictor 📊")
st.write("This app uses Machine Learning to predict if a customer will leave the company.")

# 3. Input section for user data
st.header("Enter Customer Information")

satisfaction = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charge = st.number_input("Monthly Charge ($)", min_value=0, max_value=200, value=50)

# 4. Prediction Logic
if st.button("Predict"):
    # Create input DataFrame with zeros for all columns
    input_df = pd.DataFrame(0, index=[0], columns=features)
    
    # Fill in the specific features provided by the user
    input_df['Satisfaction_Score'] = satisfaction
    input_df['Tenure_in_Months'] = tenure
    input_df['Monthly_Charge'] = monthly_charge
    
    # Get prediction
    prediction = model.predict(input_df)
    
    # 5. Display Result
    st.subheader("Results:")
    if prediction[0] == 1:
        st.error("Warning: This customer is likely to Churn (Leave)!")
    else:
        st.success("Good News: This customer is likely to Stay!")

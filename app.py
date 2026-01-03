import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Model, Features and Data
# Ensure these files are in your GitHub repository
model = joblib.load('churn_model_final.pkl')
features = joblib.load('model_columns.pkl')
df = pd.read_csv('telco.csv') 

# Page Configuration
st.set_page_config(page_title="Telco Churn AI Analytics", layout="wide", page_icon="📈")

st.title("📊 Advanced Customer Churn Analytics")
st.markdown("""
This application predicts the likelihood of customer churn and provides deep insights into customer behavior using Machine Learning.
""")

# Create Tabs for a cleaner UI
tab1, tab2 = st.tabs(["🚀 Predict Churn", "📈 Data Insights"])

with tab1:
    st.header("Individual Customer Prediction")
    st.info("Adjust the sliders and values to see the prediction.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        satisfaction = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
    with col2:
        monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, max_value=200.0, value=50.0)
        # Assuming 'Number_of_Referrals' was used in your model, otherwise you can remove this.
        referrals = st.number_input("Number of Referrals", min_value=0, max_value=20, value=0)

    if st.button("Run Prediction"):
        # Create input DataFrame
        input_data = pd.DataFrame(0, index=[0], columns=features)
        input_data['Satisfaction_Score'] = satisfaction
        input_data['Tenure_in_Months'] = tenure
        input_data['Monthly_Charge'] = monthly_charge
        
        # Get prediction and probability
        prediction = model.predict(input_data)
        
        st.subheader("Result:")
        if prediction[0] == 1:
            st.error("⚠️ High Risk: This customer is likely to CHURN.")
        else:
            st.success("✅ Low Risk: This customer is likely to STAY.")

with tab2:
    st.header("Exploratory Data Analysis (EDA)")
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Overall Churn Distribution")
        fig, ax = plt.subplots()
        # Ensure 'Customer_Status' is the correct column name in your telco.csv
        df['Customer_Status'].value_counts().plot.pie(
            autopct='%1.1f%%', 
            ax=ax, 
            colors=['#4CAF50','#FF5252'], 
            explode=(0.05, 0)
        )
        ax.set_ylabel('')
        st.pyplot(fig)

    with col4:
        st.subheader("Monthly Charges vs. Churn")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='Customer_Status', y='Monthly_Charge', data=df, ax=ax2, palette="Set2")
        st.pyplot(fig2)

    st.divider()
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10))

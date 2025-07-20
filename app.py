## importing the library
import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import plotly.express as px  # fixed alias

# loading the model
loaded_model = pickle.load(open('loan_classifier.sav', 'rb'))

# loading the dataset
loan = pd.read_csv('loan.csv')

# function for loan prediction
def loan_prediction(input_data):
    convert_data_to_numpy = np.asarray(input_data)
    reshape_input_data = convert_data_to_numpy.reshape(1, -1)
    prediction = loaded_model.predict(reshape_input_data)
    return 'Sorry your loan has not been approved' if prediction == 0 else 'Congratulations, you are eligible for the loan'

# Chart Page
def chart_page():
    st.title('Overview')

    # Creating a count plot for gender
    gig_gender = px.histogram(
        loan,
        x='Gender',
        color='Loan_Status',
        title='Gender Status of Loan Applicants',
        labels={'Loan_Status': 'Loan Status'})
    st.plotly_chart(gig_gender)  # Shows the plot

    # Adding insights
    st.subheader('Insights')
    st.markdown("The chart shows that men tend to apply for loan more than women")

# Dashboard Page
def dashboard_page():
    st.title('Dashboard Page')
    st.markdown('Input the required values')

    # Collecting user inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.selectbox('Gender (Select 0 or 1)', options=[0, 1])
        Married = st.selectbox('Married (Select 0 or 1)', options=[0, 1])
        Dependents = st.selectbox('Dependents (Select 0 or 1)', options=[0, 1])
        Education = st.selectbox('Education (Select 0 or 1)', options=[0, 1])

    with col2:
        Self_Employed = st.selectbox('Self Employed (Select 0 or 1)', options=[0, 1])
        Credit_History = st.selectbox('Credit History (Select 0 or 1)', options=[0, 1])
        Property_Area_Rural = st.selectbox('Property Area Rural (Select 0 or 1)', options=[0, 1])
        Property_Area_Urban = st.selectbox('Property Area Urban (Select 0 or 1)', options=[0, 1])

    with col3:
        ApplicantIncome = st.number_input('ApplicantIncome', value=0)
        CoapplicantIncome = st.number_input('CoapplicantIncome', value=0)
        LoanAmount = st.number_input('LoanAmount', value=0)
        Loan_Amount_Term = st.number_input('Loan_Amount_Term', value=0)

    # Click button
    if st.button('Bank Loan Application'):
        try:
            input_data = [
                int(Gender),
                int(Married),
                int(Dependents),
                int(Education),
                int(Self_Employed),
                int(Credit_History),
                int(Property_Area_Rural),
                int(Property_Area_Urban),
                float(ApplicantIncome),
                float(CoapplicantIncome),
                float(LoanAmount),
                float(Loan_Amount_Term)
            ]
            result = loan_prediction(input_data)
            st.success(result)
        except ValueError:
            st.error('Enter a valid numeric number')

# Function to switch between the tabs
def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.selectbox('Select Page', ['Chart', 'Dashboard'])

    if page == 'Chart':
        chart_page()
    elif page == 'Dashboard':
        dashboard_page()

# Run the app
if __name__ == '__main__':
    main()

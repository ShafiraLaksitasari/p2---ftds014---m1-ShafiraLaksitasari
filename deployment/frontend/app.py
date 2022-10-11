import json
import pandas as pd
import pickle
import streamlit as st
import requests

# load pipeline
pipe = pickle.load(open("model/preprocess_churn.pkl", "rb"))

st.title("Customer Churn Checker")
tenure = st.number_input("Tenure")
MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges")
SeniorCitizen = st.selectbox("Senior Citizen", ['Yes', 'No'])
Partner = st.selectbox("Partner", ['Yes', 'No'])
Dependents = st.selectbox("Dependents", ['Yes', 'No'])
OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No'])
OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No'])
DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No'])
TechSupport = st.selectbox("Tech Support", ['Yes', 'No'])
Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
PaymentMethod = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

new_data ={'tenure': tenure,
         'MonthlyCharges' : MonthlyCharges,
         'TotalCharges' : TotalCharges,
         'SeniorCitizen' :SeniorCitizen,
         'Partner' : Partner,
        'Dependents' : Dependents,
        'OnlineSecurity' : OnlineSecurity,
        'OnlineBackup' : OnlineBackup,
        'DeviceProtection' : DeviceProtection,
        'TechSupport' : TechSupport,
        'Contract' : Contract,
        'PaperlessBilling' : PaperlessBilling,
        'PaymentMethod' : PaymentMethod
}

new_data = pd.DataFrame([new_data])

# build feature
new_data = pipe.transform(new_data)
new_data = new_data.tolist()

# inference
URL = "http://backend-m-one-p-two-ocha.herokuapp.com/v1/models/churn_model:predict"
param = json.dumps({
        "signature_name":"serving_default",
        "instances":new_data
    })
r = requests.post(URL, data=param)

if r.status_code == 200:
    res = r.json()
    if res['predictions'][0][0] > 0.5:
        st.title("Churn")
    else:
        st.title("Not Churn")
else:
    st.title("Unexpected Error")
import sklearn
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import pickle
import pandas as pd
import streamlit as st

#importing encoders
with open('geo_encoder_file.pkl','rb') as file:
    geo_encoder = pickle.load(file)

with open('gender_encoder_file.pkl','rb') as file:
    gender_encoder =  pickle.load(file)

with open('input_scaler_file.pkl','rb') as file:
    input_scaler = pickle.load(file)

trained_churn_model = load_model("model.h5")

#geo_encoder.categories_ - gives the categories for which one hot encoding was performed
#gender_encoder.classes_ - gives the categories for which label encoding was performed
#defining inputs in web app with a form
st.title("Churn model prediction")
form_tab,prediction_tab = st.tabs(["Data","Prediction"])
flag=1 #cannot proceed with data preperation
with form_tab:
    st.subheader("Enter Customer Details")
    form,data = st.columns(2)
    with form:
        with st.form(key = "input_data",clear_on_submit=True): 
            customer_id= st.number_input(label="Customer ID",min_value= 0,value=None,format="%i",placeholder = "Enter valid Customer ID")
            surname = st.text_input(label="Surname",placeholder="Enter Surname/Last name")
            creditscore = st.number_input(label="Credit Score",min_value= 0,format="%i")
            geography = st.selectbox(label="Geography/Location",options=geo_encoder.categories_[0])
            gender = st.selectbox(label="Gender",options=gender_encoder.classes_)
            age= st.slider(label="Age",min_value=0,max_value=100,value=18)
            tenure=st.slider(label="Tenure",min_value=0,max_value=10)
            balance=st.number_input(label="Account Balance",min_value= 0,format="%i")
            num_of_products=st.slider(label="Number of Products",min_value=0,max_value=4,value=1)
            credit_card = st.selectbox(label="Do you have a credit card? If yes choose 1, otherwise choose 0",options = [0,1])
            active_member = st.selectbox(label="Are you a active member?If yes choose 1, otherwise choose 0",options = [0,1])
            estimated_salary = st.number_input(label="Estimated Salary",min_value= 0,format="%i")
            submitted = st.form_submit_button("Submit")
    with data:
        if submitted:
            if (age>=18 and num_of_products>=1):
                flag=0
                st.write("Customer Details are successfully submitted")
            else:
                if age<18:
                    st.write("Invalid age")
                if num_of_products<1:
                    st.write("Number of products chosen is invalid")
                st.write("Fill all the details correctly before clicking on submit")

with prediction_tab:
    st.subheader("Churn Prediction")
    if flag==0:
        #data preperation
        input_data = pd.DataFrame({
            "CustomerId":[customer_id],
            "Surname": [surname],
            "CreditScore": [creditscore],
            "Geography": [geography],
            "Gender": [gender],
            "Age": [age],	
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],	
            "HasCrCard": [credit_card],
            "IsActiveMember": [active_member],
            "EstimatedSalary": [estimated_salary]
            })
        st.write("Customer Details entered in Data tab")
        st.write(input_data)

        #dropping unecessary columns
        input_data = input_data.drop(["CustomerId","Surname"],axis=1)

        #encoding text data
        input_data["Gender"] = gender_encoder.transform([input_data['Gender']])
        geo_data = pd.DataFrame(geo_encoder.transform([input_data["Geography"]]),columns=geo_encoder.get_feature_names_out())

        #concatenating one hot encoded features
        input_data= pd.concat([input_data.drop(columns="Geography"),geo_data],axis=1)

        #scaling input data
        input_data_scaled = input_scaler.transform(input_data)

        #prediction
        prediction = trained_churn_model.predict(input_data_scaled)
        st.subheader("Will the customer exit or not?")
        st.write(f"Prediction probability: {prediction[0][0]:.2f}")
        if(prediction[0][0]>0.5):
            st.write("The customer will exit")
        else:
            st.write("The customer will not exit")
    else:
        st.write("Enter valid Customer details in Data tab to generate prediction")

        


        
    







    



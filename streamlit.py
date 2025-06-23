import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Set the page title and description
st.title("Credit Loan Eligibility Predictor")
st.write("""
This app predicts whether a loan applicant is eligible for a loan 
based on various personal and financial characteristics.
""")

# Load the pre-trained model
rf_pickle = open(r'models/model.pkl', 'rb')
rf_model = pickle.load(rf_pickle)
rf_pickle.close()


# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Health Information")
    
    bmi = st.number_input("Body Mass Index", 
                                    min_value=0, 
                                    step=0.1)
    
    smoking = st.selectbox("Smoking", options=["Yes", "No"])
     
    alcohol_drinking = st.selectbox("AlcoholDrinking",  options=["Yes", "No"])
    
    stroke = st.selectbox("Stroke", options=["Yes", "No"])
    
    physical_health = st.number_input("PhysicalHealth", 
                                    min_value=0, 
                                    step=0.1)
    
    mental_health = st.number_input("MentalHealth", 
                                    min_value=0, 
                                    step=0.1)        
     
    diff_walking = st.selectbox("DiffWalking", options=["Yes", "No"])
    
    sex = st.selectbox("Sex", options=["Male", "Female"])              
            
    age_category = st.selectbox("age_category", options=['18-24','25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'])
            
    race = st.selectbox("Race", options=['White', 'Black', 'Asian', 'Hispanic', 'American Indian/Alaskan Native', 'Other'])
    
    diabetic = st.selectbox("Diabetic", options=['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'])
    
    physical_activity = st.selectbox("PhysicalActivity", options=["Yes", "No"])
    
    genHealth = st.selectbox("GenHealth", options=['Poor', 'Fair', 'Good', 'Very good', 'Excellent'])
    
    sleep_time = st.number_input("SleepTime", 
                                    min_value=0, 
                                    step=0.5)    
    
    asthma 	= st.selectbox("Asthma", options=["Yes", "No"])
    
    kidney_disease = st.selectbox("Kidney Disease", options=["Yes", "No"])
    
    skin_cancer = st.selectbox("Skin Cancer", options=["Yes", "No"])
    
    # Submit button
    submitted = st.form_submit_button("Predict Heart Disease")


# Handle the dummy variables to pass to the model
if submitted:
    smoking = 1 if smoking=='Yes' else 0
    alcohol_drinking = 1 if alcohol_drinking=='Yes' else 0
    stroke = 1 if stroke=='Yes' else 0
    diff_walking = 1 if diff_walking=='Yes' else 0
    sex = 1 if sex=='Male' else 0
    physical_activity = 1 if physical_activity=='Yes' else 0
    asthma = 1 if asthma=='Yes' else 0
    kidney_disease = 1 if kidney_disease=='Yes' else 0
    skin_cancer = 1 if skin_cancer=='Yes' else 0
    
    age_category_18_24 = 1 if age_category == '18-24' else 0
    age_category_25_29 = 1 if age_category == '25-29' else 0
    age_category_30_34 = 1 if age_category == '30-34' else 0 
    age_category_35_39 = 1 if age_category == '18-24' else 0
    age_category_40_44 = 1 if age_category == '40-44' else 0
    age_category_45_49 = 1 if age_category == '45-49' else 0
    age_category_50_54 = 1 if age_category == '50-54' else 0  
    age_category_55_59 = 1 if age_category == '55-59' else 0 
    age_category_60_64 = 1 if age_category == '60-64' else 0 
    age_category_65_69 = 1 if age_category == '65-69' else 0 
    age_category_70_74 = 1 if age_category == '70-74' else 0
    age_category_75_79 = 1 if age_category == '75-79' else 0
    age_category_80_or_older = 1 if age_category == '80 or older' else 0
    

    race_aia_native = 1 if race == 'American Indian/Alaskan Native' else 0
    race_asian = 1 if race == 'Asian' else 0
    race_black = 1 if race == 'Black' else 0
    race_hispanic = 1 if race == 'Hispanic' else 0
    race_other = 1 if race == 'Other' else 0
    race_white = 1 if race == 'White' else 0

    gen_health_excellent = 1 if genHealth == 'Excellent' else 0
    gen_health_fair = 1 if genHealth == 'Fair' else 0
    gen_health_good = 1 if genHealth == 'Good' else 0
    gen_health_poor = 1 if genHealth == 'Poor' else 0 
    gen_health_very_good = 1 if genHealth == 'Very good' else 0

    diabetic_no = 1 if diabetic == 'No' else 0
    diabetic_no_bd = 1 if diabetic == 'No, borderline diabetes' else 0
    diabetic_yes = 1 if diabetic == 'Yes' else 0
    diabetic_yes_dp = 1 if diabetic == 'Yes (during pregnancy)' else 0


    # Prepare the input for prediction. This has to go in the same order as it was trained
    prediction_input = [[ 
                            bmi, smoking, alcohol_drinking, stroke, physical_health,
                            mental_health, diff_walking, sex, physical_activity, sleep_time,
                            asthma, kidney_disease,skin_cancer, age_category_18_24, age_category_25_29,
                            age_category_30_34, age_category_35_39, age_category_40_44, age_category_45_49, age_category_50_54,
                            age_category_55_59, age_category_60_64, age_category_65_69, age_category_70_74, age_category_75_79,
                            age_category_80_or_older, race_aia_native, race_asian, race_black, race_hispanic,
                            race_other, race_white, gen_health_excellent, gen_health_fair, gen_health_good,
                            gen_health_poor, gen_health_very_good, diabetic_no, diabetic_no_bd, diabetic_yes,
                            diabetic_yes_dp
                        ]]

    # Make prediction
    new_prediction = rf_model.predict(prediction_input)

    # Display result
    st.subheader("Prediction Result:")
    if new_prediction[0] == '1':
        st.write("You may have heart disease.")
    else:
        st.write("You may not have heart disease.")

st.write(
    """We used a machine learning (Random Forest) model to predict the probobility of heart disease, the features used in this prediction are ranked by relative
    importance below."""
)
# st.image("feature_importance.png")

# website: https://assignment-kmeans.streamlit.app/

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set the page title and description
st.title("Diamond Clustering and Quality Analysis using K-Means")
st.write("""
            An unsupervised machine learning project using real-world diamond pricing data
        """)

# Load the pre-trained model
try:
    with open('models/model.pkl', 'rb') as f:
        kmodel = pickle.load(f)    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("Model file not found!")
    st.error("Model file is missing.")
except Exception as e:
    logging.exception("Unexpected error occurred while loading model.")
    st.error("Something went wrong while loading the model.")



# Prepare the form to collect user inputs
with st.form("user_inputs"):
    st.subheader("Diamond Properties")
    
    carat = st.number_input("Carat", 
                                    min_value=0.0, 
                                    step=0.05, 
                                    value=1.0)
    
    cut = st.selectbox("Cut", options=['Ideal', 'Premium', 'Good', 'Very Good', 'Fair'])
    
    color = st.selectbox("Color", options=['E', 'I', 'J', 'H', 'F', 'G', 'D'])
     
    clarity = st.selectbox("Clarity",  options=['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF'])
    
    depth = st.number_input("Depth(%)", 
                                    min_value=0.0, 
                                    step=1.0, 
                                    value=50.0)
    
    table = st.number_input("Table(%)", 
                                    min_value=0.0, 
                                    step=1.0, 
                                    value=50.0)       
     
    x = st.number_input("x(mm)", 
                            min_value=0.0, 
                            step=1.0, 
                            value=10.0)       
    
    y = st.number_input("y(mm)", 
                            min_value=0.0, 
                            step=1.0, 
                            value=10.0)        
    
    z = st.number_input("z(mm)", 
                            min_value=0.0, 
                            step=1.0, 
                            value=10.0)   
       
    # Submit button
    submitted = st.form_submit_button("Start Analysis")


# Handle the dummy variables to pass to the model
if submitted:

    cut_map = {'Fair':1, 'Good':2, 'Very Good':3, 'Premium':4, 'Ideal':5}
    color_map = {'J':1, 'I':2, 'H':3, 'G':4, 'F':5, 'E':6, 'D':7}
    clarity_map = {
                    'I1': 1,
                    'SI2': 2,
                    'SI1': 3,
                    'VS2': 4,
                    'VS1': 5,
                    'VVS2': 6,
                    'VVS1': 7,
                    'IF': 8
                    }
    cut=cut_map[cut]
    color=color_map[color]
    clarity=clarity_map[clarity]
    
    # Construct the input in the same feature order used during training
    prediction_input = [[ 
                            carat,cut,color,
                            clarity,depth,table,
                            x,y,z
                        ]]


    
    # Make prediction
    input_scaled = scaler.transform(prediction_input)
    new_prediction=kmodel.predict(input_scaled)

    # print(f'{prediction_input}\n')
    # print(f'{input_scaled}\n')
    # print(f'{new_prediction[0]}\n')


    # Display result
    st.subheader("Prediction Result:")
    diamond_dict= {
                        0: "Cluster 0: Mid-size diamonds with balanced quality.",
                        1: "Cluster 1: Small, high-quality diamonds with premium cut and clarity.",
                        2: "Cluster 2: Large diamonds that may compromise on clarity and cut."
                    }
    
    st.write(diamond_dict[new_prediction[0]])

st.write(
    """We used an unsupervised machine learning model (KMeans clustering) to group 
       diamonds based on their physical and quality features. The clusters represent 
       different diamond types with similar characteristics."""
)


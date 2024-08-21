import pickle

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import warnings
warnings.filterwarnings("ignore")

random.seed(42)
#picle file to use the previously saved state of the model trained
pickle_in = open("application_files/svr_model.pkl", "rb")
svr_model = pickle.load(pickle_in)

#Normalizing the data input the user
def normalize_feature(value, min_val, max_val):
    return (float(value) - min_val) / (max_val - min_val)

#De-normalizing the predictions to output it to the user in original form
def denormalize_price(normalized_price, min_val, max_val):
    return normalized_price * (max_val - min_val) + min_val

#One hot encoding of categorical values of property type
def encode_property_type(property_type):
    type_h, type_u, type_t = 0, 0, 0
    if property_type == "House/Villa":
        type_h = 1
    elif property_type == "Duplex":
        type_u = 1
    elif property_type == "Townhouse":
        type_t = 1
    return type_h, type_u, type_t

#prediction model with normalized features
def house_price_prediction(rooms, distance, bathroom, car, propertycount, type_h, type_u, type_t):
    # Normalize each feature
    normalized_rooms = normalize_feature(rooms, 1, 10)
    normalized_distance = normalize_feature(distance, 0.0, 47.3)
    normalized_bathroom = normalize_feature(bathroom, 0, 8)
    normalized_car = normalize_feature(car, 0.0, 10.0)
    normalized_propertycount = normalize_feature(propertycount, 389, 21650)
    normalized_type_h = normalize_feature(type_h, 0, 1)
    normalized_type_u = normalize_feature(type_u, 0, 1)
    normalized_type_t = normalize_feature(type_t, 0, 1)
# Model call for prediction
    normalized_prediction = svr_model.predict([[normalized_rooms, normalized_distance, normalized_bathroom, normalized_car, normalized_propertycount, normalized_type_h, type_u, type_t]])
    prediction = denormalize_price(normalized_prediction, 136.0, 710000.0)
    return (prediction)
""" 
Code for the webpage UI design
"""

#Title of the UI interface of the product
st.set_page_config(
    page_title="Melbourne Housing Price Prediction",
    # page_icon=":house:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
        body {
            background-color: #f5f5f5;
        }
        .stApp {
            background-color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

img_dataset_cover = Image.open("application_files/webpage-cover.jpg")

#Website description
with st.container():
    st.image(img_dataset_cover, use_column_width=True)
    st.title("Melbourne Housing Price Prediction")
    st.write(
        "Hello, this website is designed to predict the house price based on the inputs given by the user."
    )

    
with st.container():
    st.write("---")
    st.header("Please enter your house requirement")    

def reset_fields():
    st.session_state.my_text_input1 = ""
    st.session_state.my_text_input2 = ""
    st.session_state.my_text_input3 = ""
    st.session_state.my_text_input4 = ""
    st.session_state.my_text_input5 = ""
    st.session_state.my_text_input6 = ""
    
#Radio button to fetch the property type
propertytype = st.radio("Select the Property Type", options=["House/Villa", "Duplex", "Townhouse"], horizontal=True)

if 'last_reset_option' not in st.session_state:
    st.session_state.last_reset_option = None

if st.session_state.last_reset_option != propertytype:
    st.session_state.last_reset_option = propertytype
    reset_fields()
    
col1, col2 = st.columns(2)
#Implementation of the input fields in the UI to take input from the user   
with col1:
    rooms = st.text_input("Rooms",placeholder="Number of rooms: Please enter numerical value. Ex: 2", key = 'my_text_input1')
    bathroom = st.text_input("Bathroom",placeholder="Number of Bathrooms: Please enter numerical value. Ex: 2", key = 'my_text_input2')
    car = st.text_input("Car",placeholder="Number of carspots: Please enter numerical value. Ex: 2", key = 'my_text_input3')
    
with col2:
    distance = st.text_input("Distance",placeholder="Please enter numerical value. Ex: 10.5", key = 'my_text_input4')
    propertycount = st.text_input("Property Count",placeholder="Number of properties that exist in the suburb: Please enter numerical value. Ex: 4019", key = 'my_text_input5')
    area = st.text_input("Area",placeholder="Land Size: Please enter numerical value. Ex: 250", key = 'my_text_input6')


type_h, type_u, type_t = encode_property_type(propertytype)

#UI design for the Prediction by the model on the webpage 
result = ""
predictions = []

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    pass
with col2:
    pass
with col4:
    pass
with col4:
    predict_button = st.button("Predict House Price", type="primary")
if predict_button:
    prediction = house_price_prediction(rooms, distance, bathroom, car, propertycount, type_h, type_u, type_t)
    result = np.round(abs(prediction * float(area)), 3)
    predictions.append({"Rooms": rooms, "Distance": float(distance), "Area": float(area), "Price": result})
    st.success('The House Price is: ${}'.format(result))

    st.write("House Prices for varying range of inputs:")
# Recommendations based on the varying range of user input
    for variation_room in range(-1, 2):
        for variation_distance in range(-1, 3):
            for variation_area in range(-2, 3):
                rooms_variation = int(rooms) + variation_room
                distance_variation = float(distance) + (variation_distance * 2)
                area_variation = float(area) + (variation_area * 10)
                prediction = house_price_prediction(rooms_variation, distance_variation, bathroom, car, propertycount, type_h, type_u, type_t)
                result = np.round(abs(prediction * float(area_variation)), 3)
                predictions.append({"Rooms": rooms_variation, "Distance": float(distance_variation), "Area": float(area_variation), "Price": result})


if predictions:
    df_predictions = pd.DataFrame(predictions)    
    df_predictions['Area'] = df_predictions['Area'].astype(float)
    df_predictions['Price'] = df_predictions['Price'].astype(float)

    # Plot the results using Plotly on the webpage
    fig = px.scatter(df_predictions, x='Area', y='Price', color='Price', size='Price', opacity=0.7,
                     title='House Price Predictions',
                     labels={'Area': 'Area', 'Price': 'Predicted Price'},
                     size_max=15, hover_data=['Rooms', 'Distance'])
    fig.add_traces(
        px.scatter(x=[df_predictions.iloc[0,2]], y=[df_predictions.iloc[0,3]]).update_traces(marker_size=15, marker_color="red").data[0]
    )
    fig.update_layout(width=600, height=600)
    st.plotly_chart(fig, use_container_width=True)

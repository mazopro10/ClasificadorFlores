
import streamlit as st
import pandas as pd
import joblib

# --- Load the pre-trained model and LabelEncoder ---
# In a real Streamlit app, you would typically load these at the start
# so they are cached and not reloaded on every interaction.
# Assuming the joblib files are in the same directory as the app.

# If running in Colab and the files are in /content/,
# you might need to adjust the paths if the .py file is saved elsewhere.

try:
    mlp_model = joblib.load('mlp_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    st.error("Error: Model or LabelEncoder files not found. Make sure 'mlp_model.joblib' and 'label_encoder.joblib' are in the same directory as this script.")
    st.stop() # Stop the app if files are not found

# --- Streamlit UI ---
st.title('Flower Species Predictor')
st.write('Enter the measurements of the flower to predict its species.')

# Input fields for features
st.sidebar.header('Flower Measurements (cm)')
sepal_length = st.sidebar.slider('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.sidebar.slider('Sepal Width (cm)', min_value=0.0, max_value=5.0, value=3.0, step=0.1)
petal_length = st.sidebar.slider('Petal Length (cm)', min_value=0.0, max_value=7.0, value=4.0, step=0.1)
petal_width = st.sidebar.slider('Petal Width (cm)', min_value=0.0, max_value=3.0, value=1.0, step=0.1)

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

# Prediction button
if st.button('Predict Species'):
    if mlp_model and label_encoder:
        # Make prediction
        prediction = mlp_model.predict(input_data)

        # Decode the prediction
        decoded_prediction = label_encoder.inverse_transform(prediction)

        st.success(f'The predicted species is: {decoded_prediction[0]}')
    else:
        st.error("Model or LabelEncoder not loaded. Please check the file paths.")

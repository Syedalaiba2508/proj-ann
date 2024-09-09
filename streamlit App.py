import streamlit as st
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load the trained model (ensure you have the trained model saved in the same directory)
@st.cache_resource
def load_model():
    model = keras.models.load_model('digit_ann_model.h5')
    return model

model = load_model()

# Create a user interface for inputting digit data
st.title("Digit Classification using ANN")

st.write("Input pixel data of the digit to predict its label:")

# Create sliders for the 64 features (corresponding to 8x8 pixel grid)
input_data = []
for i in range(64):
    value = st.slider(f'Pixel {i + 1}', 0.0, 16.0, 8.0)
    input_data.append(value)

# Convert the input into a NumPy array
input_data = np.array(input_data).reshape(1, -1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict using the trained model
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    predicted_label = np.argmax(prediction)
    st.write(f"The predicted label is: {predicted_label}")


import pickle
import streamlit as st
import numpy as np

# Title of the application
st.title('Bank account creation prediction')

# Load the trained machine learning model
def load_model():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model
model = load_model()

# Define features variable outside of the prediction button scope
features = []

# Input fields for features
feature1 = st.text_input('education_level', 0, 10)
feature2 = st.slider('cellphone_access', 0, 5)
feature3 = st.text_input('gender_of_respondent', 0, 2)
feature4 = st.text_input('year, 20000, 2020 ')

# Validation button
if st.button('Validate'):
    # Validation logic
    st.write('Feature 1:', feature1)
    st.write('Feature 2:', feature2)
    st.write('Feature 3:', feature3)
    st.write('Feature 4:', feature4)
    st.write('Validation complete!')

# Define features variable outside of the prediction button scope
features = []

# Prediction button
if st.button('Predict'):
  features.append(float(feature1))
  features.append(feature2)
  features.append(feature3)
  features.append(feature4)

  # Make predictions
  prediction = model.predict(np.array(features).reshape(1, -1))

  # Display prediction
  st.write('Prediction:', prediction)

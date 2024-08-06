import streamlit as st
import joblib

# Load your model
model = joblib.load('wine_quality_prediction.pkl')

# Define the labels
quality_labels = {
    0: 'BAD QUALITY WINE',
    1: 'POOR QUALITY WINE',
    2: 'AVERAGE QUALITY WINE',
    3: 'GOOD QUALITY WINE',
    4: 'EXCELLENT QUALITY WINE'
    # Adjust the labels based on your model's output range
}


def predict_wine_quality(feat):
    """Function to predict wine quality based on input features"""
    pred = model.predict([feat])
    # Convert numeric prediction to a descriptive label
    return quality_labels.get(pred[0], 'UNKNOWN QUALITY WINE')


st.title('Wine Quality Prediction')
st.write('Enter the features of the wine to predict its quality.')

# Define the input features
fixed_acidity = st.text_input('Fixed Acidity', value='0.0')
volatile_acidity = st.text_input('Volatile Acidity', value='0.0')
citric_acid = st.text_input('Citric Acid', value='0.0')
residual_sugar = st.text_input('Residual Sugar', value='0.0')
chlorides = st.text_input('Chlorides', value='0.0')
free_sulfur_dioxide = st.text_input('Free Sulfur Dioxide', value='0')
total_sulfur_dioxide = st.text_input('Total Sulfur Dioxide', value='0')
density = st.text_input('Density', value='0.99')
pH = st.text_input('pH', value='2.5')
sulphates = st.text_input('Sulphates', value='0.0')
alcohol = st.text_input('Alcohol', value='8.0')

# Convert inputs to appropriate types
try:
    features = [
        float(fixed_acidity),
        float(volatile_acidity),
        float(citric_acid),
        float(residual_sugar),
        float(chlorides),
        int(free_sulfur_dioxide),
        int(total_sulfur_dioxide),
        float(density),
        float(pH),
        float(sulphates),
        float(alcohol)
    ]
except ValueError:
    st.write('Please enter valid numeric values for all inputs.')
    features = None

if st.button('Predict') and features:
    prediction = predict_wine_quality(features)
    st.write(f'{prediction}')

import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model and data files
model = joblib.load('model.pkl')
feature_names = joblib.load('feature_names.pkl')
index_to_disease = joblib.load('index_to_disease.pkl')

# Function to make predictions based on symptoms
def predict_disease(symptoms):
    symptoms_df = pd.DataFrame([symptoms], columns=feature_names)
    
    symptoms_df = symptoms_df.reindex(columns=feature_names, fill_value=0)
    
    prediction_index = model.predict(symptoms_df)[0]
    
    disease_name = index_to_disease.get(prediction_index, 'Desconocida')
    
    return disease_name

st.title("Vector Borne Disease Prediction")

# List of symptom labels
symptom_labels = ['sudden_fever', 'headache', 'mouth_bleed', 'nose_bleed', 'muscle_pain',
                  'joint_pain', 'vomiting', 'rash', 'diarrhea', 'hypotension', 'pleural_effusion',
                  'ascites', 'gastro_bleeding', 'swelling', 'nausea', 'chills', 'myalgia',
                  'digestion_trouble', 'fatigue', 'skin_lesions', 'stomach_pain', 'orbital_pain',
                  'neck_pain', 'weakness', 'back_pain', 'weight_loss', 'gum_bleed', 'jaundice',
                  'coma', 'diziness', 'inflammation', 'red_eyes', 'loss_of_appetite', 'urination_loss',
                  'slow_heart_rate', 'abdominal_pain', 'light_sensitivity', 'yellow_skin', 'yellow_eyes',
                  'facial_distortion', 'microcephaly', 'rigor', 'bitter_tongue', 'convulsion', 'anemia',
                  'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia', 'stiff_neck',
                  'irritability', 'confusion', 'tremor', 'paralysis', 'lymph_swells', 'breathing_restriction',
                  'toe_inflammation', 'finger_inflammation', 'lips_irritation', 'itchiness', 'ulcers',
                  'toenail_loss', 'speech_problem', 'bullseye_rash']

# Create 3 columns to display the symptom checkboxes
col1, col2, col3 = st.columns(3)

with col1:
    for label in symptom_labels[:len(symptom_labels)//3]:
        globals()[label] = st.checkbox(label.replace('_', ' ').title())

with col2:
    for label in symptom_labels[len(symptom_labels)//3:2*len(symptom_labels)//3]:
        globals()[label] = st.checkbox(label.replace('_', ' ').title())

with col3:
    for label in symptom_labels[2*len(symptom_labels)//3:]:
        globals()[label] = st.checkbox(label.replace('_', ' ').title())

# Convert the selected symptoms (True/False) to a list of integers (1 for True, 0 for False)
symptoms = [int(globals()[label]) for label in symptom_labels]

# Create a button that, when clicked, triggers the disease prediction
if st.button('Predecir Enfermedad'):
    disease_name = predict_disease(symptoms)
    st.write(f"La enfermedad predicha es: {disease_name}")

# app.py (Streamlit App Code)
import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('disease_features.csv')
    
    # Add Category column
    categories = {
        'Acute Coronary Syndrome': 'cardiovascular',
        'Adrenal Insufficiency': 'endocrine',
        'Alzheimer': 'neurological',
        'Aortic Dissection': 'cardiovascular',
        'Asthma': 'respiratory',
        'Atrial Fibrillation': 'cardiovascular',
        'Cardiomyopathy': 'cardiovascular',
        'COPD': 'respiratory',
        'Diabetes': 'endocrine',
        'Epilepsy': 'neurological',
        'Gastritis': 'gastrointestinal',
        'Gastro-oesophageal Reflux Disease': 'gastrointestinal',
        'Heart Failure': 'cardiovascular',
        'Hyperlipidemia': 'cardiovascular',
        'Hypertension': 'cardiovascular',
        'Migraine': 'neurological',
        'Multiple Sclerosis': 'neurological',
        'Peptic Ulcer Disease': 'gastrointestinal',
        'Pituitary Disease': 'endocrine',
        'Pneumonia': 'respiratory',
        'Pulmonary Embolism': 'cardiovascular',
        'Stroke': 'neurological',
        'Thyroid Disease': 'endocrine',
        'Tuberculosis': 'respiratory',
        'Upper Gastrointestinal Bleeding': 'gastrointestinal'
    }
    df['Category'] = df['Disease'].map(categories)
    return df

df = load_data()

# Preprocess
def parse_and_join(text):
    return ' '.join(eval(text))

# Clean risk factors
def clean_risk_factor(rf_list):
    cleaned = []
    for rf in eval(rf_list):
        rf = rf.replace('etc.', '').replace('(', '').replace(')', '').replace(';', '').replace(',', '').strip()
        if 'Genetic syndromes' in rf:
            rf = 'Genetic syndromes'
        if 'Close contact' in rf:
            rf = 'Close contact'
        if 'Areas with high TB prevalence' in rf:
            rf = 'Areas with high TB prevalence'
        if 'Genetic Abnormalities' in rf:
            rf = 'Genetic Abnormalities'
        if 'Genetic and environmental factors' in rf:
            rf = 'Genetic and environmental factors'
        if 'Certain Medications' in rf:
            rf = 'Certain Medications'
        if 'Certain medications' in rf:
            rf = 'Certain medications'
        if 'Metabolic Abnormalities' in rf:
            rf = 'Metabolic Abnormalities'
        if 'Other  Co-administration' in rf:
            rf = 'Co-administration of corticosteroids and bisphosphonates with NSAIDs'
        if 'Other Factors' in rf:
            rf = 'Other Factors'
        if 'Socioeconomic factors' in rf:
            rf = 'Socioeconomic factors'
        if 'medical conditions' in rf:
            rf = 'medical conditions'
        if 'Location of the lesion' in rf:
            rf = 'Location of the lesion'
        if 'Long-term exposure to harmful particles' in rf:
            rf = 'Long-term exposure to harmful particles'
        cleaned.append(rf)
    return cleaned

# Apply preprocessing
df['Risk_Factors_str'] = df['Risk Factors'].apply(parse_and_join)
df['Symptoms_str'] = df['Symptoms'].apply(parse_and_join)
df['Signs_str'] = df['Signs'].apply(parse_and_join)

# TF-IDF Vectorization
tfidf_rf = TfidfVectorizer()
tfidf_sym = TfidfVectorizer()
tfidf_sign = TfidfVectorizer()

X_rf = tfidf_rf.fit_transform(df['Risk_Factors_str'])
X_sym = tfidf_sym.fit_transform(df['Symptoms_str'])
X_sign = tfidf_sign.fit_transform(df['Signs_str'])
X_tfidf = hstack([X_rf, X_sym, X_sign])
y = df['Category']

# Streamlit app
st.title("Disease Category Predictor")

# User inputs
k = st.slider("Select k for KNN", 3, 5, 3)
metric = st.selectbox("Select Distance Metric", ['euclidean', 'manhattan', 'cosine'])

rf = st.text_input("Risk Factors (comma-separated)", "hypertension, smoking")
sym = st.text_input("Symptoms (comma-separated)", "chest pain, dyspnea")
sign = st.text_input("Signs (comma-separated)", "tachycardia")

# Train model
knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
knn.fit(X_tfidf, y)

# Prediction
if st.button("Predict"):
    rf_str = ' '.join(rf.split(','))
    sym_str = ' '.join(sym.split(','))
    sign_str = ' '.join(sign.split(','))
    
    X_rf_new = tfidf_rf.transform([rf_str])
    X_sym_new = tfidf_sym.transform([sym_str])
    X_sign_new = tfidf_sign.transform([sign_str])
    X_new = hstack([X_rf_new, X_sym_new, X_sign_new])
    
    pred = knn.predict(X_new)
    st.write(f"Predicted Category: {pred[0]}")
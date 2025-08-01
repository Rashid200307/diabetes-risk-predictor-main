import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- PATH CONFIGURATION FOR STREAMLIT CLOUD ---
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'models', 'xgboost.pkl')
FEATURE_IMG_PATH = os.path.join(current_dir, 'models', 'feature_importance.png')

# --- ERROR HANDLING FOR DEPENDENCIES ---
try:
    # Import XGBoost only when needed
    from xgboost import XGBClassifier
except ImportError:
    st.error("""
    XGBoost not installed! Please add 'xgboost==2.0.3' to requirements.txt.
    """)
    st.stop()

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {str(e)}")
    st.stop()

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# Language toggle
lang = st.sidebar.radio("Language", ["English", "Malay"])

# App title
titles = {"English": "Diabetes Risk Prediction", "Malay": "Ramalan Risiko Kencing Manis"}
st.title(titles[lang])

# --- INPUT FORM ---
input_labels = {
    "English": {
        "Pregnancies": "Number of Pregnancies",
        "Glucose": "Glucose Level (mg/dL)",
        "BloodPressure": "Blood Pressure (mmHg)",
        "SkinThickness": "Skin Thickness (mm)",
        "Insulin": "Insulin Level (μU/mL)",
        "BMI": "Body Mass Index (BMI)",
        "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
        "Age": "Age (years)"
    },
    "Malay": {
        "Pregnancies": "Bilangan Kehamilan",
        "Glucose": "Tahap Glukosa (mg/dL)",
        "BloodPressure": "Tekanan Darah (mmHg)",
        "SkinThickness": "Ketebalan Kulit (mm)",
        "Insulin": "Tahap Insulin (μU/mL)",
        "BMI": "Indeks Jisim Badan (BMI)",
        "DiabetesPedigreeFunction": "Fungsi Salasilah Diabetes",
        "Age": "Umur (tahun)"
    }
}

inputs = {}
with st.form("prediction_form"):
    cols = st.columns(2)
    with cols[0]:
        inputs['Pregnancies'] = st.slider(input_labels[lang]["Pregnancies"], 0, 17, 1)
        inputs['Glucose'] = st.slider(input_labels[lang]["Glucose"], 0, 200, 100)
        inputs['BloodPressure'] = st.slider(input_labels[lang]["BloodPressure"], 0, 122, 70)
        inputs['SkinThickness'] = st.slider(input_labels[lang]["SkinThickness"], 0, 99, 20)
        
    with cols[1]:
        inputs['Insulin'] = st.slider(input_labels[lang]["Insulin"], 0, 846, 79)
        inputs['BMI'] = st.slider(input_labels[lang]["BMI"], 0.0, 67.1, 25.0)
        inputs['DiabetesPedigreeFunction'] = st.slider(input_labels[lang]["DiabetesPedigreeFunction"], 0.08, 2.42, 0.47)
        inputs['Age'] = st.slider(input_labels[lang]["Age"], 21, 81, 30)
    
    predict_text = {"English": "Predict Risk", "Malay": "Ramal Risiko"}
    submitted = st.form_submit_button(predict_text[lang])

# --- PREDICTION LOGIC ---
if submitted:
    with st.spinner("Analyzing your risk..." if lang == "English" else "Menganalisis risiko anda..."):
        # Create dataframe from inputs
        input_df = pd.DataFrame([inputs])
        
        # Make prediction
        try:
            risk = model.predict_proba(input_df)[0][1] * 100
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.stop()
        
        # Display results
        st.subheader("Prediction Results" if lang == "English" else "Keputusan Ramalan")
        
        # Risk visualization
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.barh(['Risk Level'], [risk], color='#ff6b6b' if risk > 50 else '#51cf66')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Risk Percentage' if lang == "English" else "Peratusan Risiko")
        ax.set_facecolor('none')  # Transparent background
        fig.patch.set_facecolor('none')
        ax.tick_params(colors=('white' if st.get_option("theme.backgroundColor") == '#0e1117' else 'black'))
        st.pyplot(fig, transparent=True)
        
        # Risk interpretation
        if risk < 30:
            message = "✅ Low Risk: Maintain healthy diet and exercise"
            malay_message = "✅ Risiko Rendah: Kekalkan diet sihat dan bersenam"
            color = "green"
        elif risk < 70:
            message = "⚠️ Medium Risk: Reduce sugar intake, monitor glucose monthly"
            malay_message = "⚠️ Risiko Sederhana: Kurangkan pengambilan gula, pantau glukosa bulanan"
            color = "orange"
        else:
            message = "❌ High Risk: Consult doctor immediately, start medication"
            malay_message = "❌ Risiko Tinggi: Berjumpa doktor segera, mulakan ubat-ubatan"
            color = "red"
        
        st.markdown(
            f"<p style='font-size:20px; color:{color}; text-align:center;'>{message if lang == 'English' else malay_message}</p>", 
            unsafe_allow_html=True
        )
        
        # Prevention tips
        st.subheader("Prevention Tips" if lang == "English" else "Tip Pencegahan")
        tips = {
            "English": [
                "🍎 Maintain a balanced diet with low sugar intake",
                "🏃‍♂️ Exercise at least 30 minutes daily",
                "🩸 Regularly monitor your blood glucose levels",
                "🚭 Avoid smoking and limit alcohol consumption",
                "😴 Get 7-8 hours of quality sleep each night",
                "🥦 Increase fiber intake with vegetables and whole grains",
                "💧 Stay hydrated with water instead of sugary drinks"
            ],
            "Malay": [
                "🍎 Mengekalkan diet seimbang dengan pengambilan gula yang rendah",
                "🏃‍♂️ Bersenam sekurang-kurangnya 30 minit setiap hari",
                "🩸 Pantau tahap glukosa darah anda secara berkala",
                "🚭 Elakkan merokok dan hadkan pengambilan alkohol",
                "😴 Dapatkan tidur berkualiti 7-8 jam setiap malam",
                "🥦 Tingkatkan pengambilan serat dengan sayur-sayuran dan bijirin penuh",
                "💧 Minum air secukupnya menggantikan minuman bergula"
            ]
        }
        
        for tip in tips[lang]:
            st.info(tip)

# --- SIDEBAR FEATURES ---
st.sidebar.divider()
st.sidebar.subheader("About" if lang == "English" else "Mengenai")
st.sidebar.info("""
This app predicts diabetes risk using machine learning. 
It analyzes health parameters to assess your risk level.
""" if lang == "English" else """
Aplikasi ini meramalkan risiko kencing manis menggunakan pembelajaran mesin. 
Ia menganalisis parameter kesihatan untuk menilai tahap risiko anda.
""")

# Feature importance visualization
st.sidebar.divider()
if st.sidebar.checkbox("Show Feature Importance" if lang == "English" else "Tunjukkan Kepentingan Ciri"):
    try:
        st.subheader("Feature Importance" if lang == "English" else "Kepentingan Ciri")
        st.image(FEATURE_IMG_PATH, use_column_width=True)
        st.caption("How different health factors contribute to diabetes risk" if lang == "English" else 
                   "Bagaimana faktor kesihatan berbeza menyumbang kepada risiko kencing manis")
    except Exception as e:
        st.warning(f"Feature importance image not found: {str(e)}")

# --- FOOTER ---
st.divider()
st.caption("""
Developed for BIT4333 Introduction to Machine Learning | 
[GitHub Repository](https://github.com/yourusername/diabetes-ml-project)
""" if lang == "English" else """
Dibangunkan untuk BIT4333 Pengenalan Kepada Pembelajaran Mesin | 
[Repositori GitHub](https://github.com/yourusername/diabetes-ml-project)
""")

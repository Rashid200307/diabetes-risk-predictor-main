import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- PATH CONFIGURATION FOR STREAMLIT CLOUD ---
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'models', 'xgboost.pkl')
FEATURE_IMG_PATH = os.path.join(current_dir, 'models', 'feature_importance.png')
PERFORMANCE_REPORT_PATH = os.path.join(current_dir, 'models', 'performance_report.md')

# --- ERROR HANDLING FOR DEPENDENCIES ---
try:
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

# --- MODEL PERFORMANCE METRICS ---
MODEL_PERFORMANCE = {
    "Accuracy": 0.78,
    "F1 Score": 0.75,
    "ROC AUC": 0.85,
    "Precision": 0.72,
    "Recall": 0.78
}

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS with dark/light mode support
st.markdown("""
<style>
    :root {
        /* Light theme colors */
        --primary-color: #3A86FF;
        --secondary-color: #00E676;
        --background-color: #F5F7FA;
        --card-color: #FFFFFF;
        --text-color: #333333;
        --light-gray: #F0F2F6;
        --medium-gray: #E0E0E0;
        --dark-gray: #757575;
        --success-color: #2ED573;
        --warning-color: #FFCC00;
        --danger-color: #FF4757;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            /* Dark theme colors */
            --primary-color: #3A86FF;
            --secondary-color: #00E676;
            --background-color: #1E1E1E;
            --card-color: #2D2D2D;
            --text-color: #F5F5F5;
            --light-gray: #2D2D2D;
            --medium-gray: #444444;
            --dark-gray: #AAAAAA;
            --success-color: #2ED573;
            --warning-color: #FFCC00;
            --danger-color: #FF4757;
        }
    }

    /* Base styles */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: var(--text-color);
        background-color: var(--background-color);
        line-height: 1.6;
    }

    /* Layout improvements */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Header styles */
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-color);
        margin-bottom: 15px;
    }

    h1 {
        font-size: 2.2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    h2 {
        font-size: 1.8rem;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 5px;
        margin-top: 2rem;
    }

    h3 {
        font-size: 1.4rem;
        margin-top: 1.5rem;
    }

    /* Card styles */
    .card {
        background-color: var(--card-color);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 2rem;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--medium-gray);
    }

    /* Button styles */
    .stButton>button {
        background-color: var(--primary-color);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }

    .stButton>button:hover {
        background-color: #2A6FD9;
        transform: translateY(-2px);
    }

    /* Input styles */
    .stSlider, .stNumberInput, .stTextInput, .stSelectbox {
        margin-bottom: 20px;
    }

    /* Divider styles */
    .stDivider {
        background: linear-gradient(to right, transparent, var(--medium-gray), transparent) !important;
        height: 1px !important;
        margin: 2rem 0 !important;
    }

    /* Metric card styles */
    .st-emotion-cache-7ym5gk {
        background-color: var(--card-color) !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }

    /* Risk container */
    .risk-container {
        margin: 2rem 0;
        background-color: var(--card-color);
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.05);
    }

    /* Feature importance container */
    .feature-importance-container {
        background-color: var(--card-color);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    }

    /* Alert styles */
    .stAlert {
        border-radius: 8px !important;
        padding: 15px !important;
        margin-bottom: 1rem !important;
    }

    /* Sidebar styles */
    .st-emotion-cache-1v0mbdj {
        background-color: var(--card-color) !important;
        border-right: 1px solid var(--medium-gray) !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: var(--primary-color) !important;
    }

    /* Markdown adjustments */
    .stMarkdown {
        margin-bottom: 1rem;
    }

    /* Slider label styles */
    .stSlider label {
        font-weight: 500 !important;
        color: var(--text-color) !important;
    }

    /* Input container styles */
    .st-emotion-cache-1offfwp {
        background-color: var(--light-gray) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }

    /* Selectbox styles */
    .st-emotion-cache-9ycgxx {
        background-color: var(--card-color) !important;
        border: 1px solid var(--medium-gray) !important;
    }

    /* Make the entire app container have proper background */
    .stApp {
        background-color: var(--background-color) !important;
    }

    /* Fix for dark mode text inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: var(--card-color) !important;
        color: var(--text-color) !important;
    }

    /* Custom container for the main content */
    .main-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 20px;
    }

    /* Header section */
    .header-section {
        text-align: center;
        margin-bottom: 2rem;
        padding: 0 1rem;
    }

    .header-section h1 {
        font-size: 2.5rem;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }

    .header-section p {
        font-size: 1.1rem;
        color: var(--dark-gray);
    }

    /* Responsive grid for tips */
    .tips-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 15px;
        margin-top: 1rem;
    }

    .tip-card {
        background-color: rgba(58, 134, 255, 0.05);
        border-left: 4px solid var(--primary-color);
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }

    .tip-icon {
        font-size: 1.5rem;
        margin-right: 10px;
    }

    .tip-title {
        margin: 0;
        color: var(--primary-color);
        font-size: 1rem;
        font-weight: bold;
    }

    .tip-text {
        margin: 0;
        font-size: 0.95rem;
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# Language toggle
lang = st.sidebar.radio("Language", ["English", "Malay"], key="language_selector")

# App title with improved header section
st.markdown('''
<div class="header-section">
    <h1>🩺 Diabetes Risk Prediction</h1>
    <p>Assess your diabetes risk based on health metrics</p>
</div>
''', unsafe_allow_html=True)

if lang == "Malay":
    st.markdown('''
    <div class="header-section">
        <h1>🩺 Ramalan Risiko Kencing Manis</h1>
        <p>Menilai risiko kencing manis berdasarkan metrik kesihatan</p>
    </div>
    ''', unsafe_allow_html=True)

# --- MODEL TRUST SECTION ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("About Our Model" if lang == "English" else "Tentang Model Kami")

# Performance metrics in columns with improved layout
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{MODEL_PERFORMANCE['Accuracy']*100:.1f}%")
with col2:
    st.metric("ROC AUC", f"{MODEL_PERFORMANCE['ROC AUC']:.3f}")
with col3:
    st.metric("F1 Score", f"{MODEL_PERFORMANCE['F1 Score']:.3f}")
with col4:
    st.metric("Recall", f"{MODEL_PERFORMANCE['Recall']*100:.1f}%")

# Model description with better formatting
st.markdown("""
<div style="background-color: rgba(58, 134, 255, 0.1); border-left: 4px solid var(--primary-color); padding: 15px; border-radius: 0 8px 8px 0; margin: 15px 0;">
<p style="margin: 0;">
Our machine learning model was trained on a comprehensive dataset of medical records and has been rigorously validated for accuracy. It uses the XGBoost algorithm, which is known for its high performance in medical prediction tasks.
</p>
</div>
""" if lang == "English" else """
<div style="background-color: rgba(58, 134, 255, 0.1); border-left: 4px solid var(--primary-color); padding: 15px; border-radius: 0 8px 8px 0; margin: 15px 0;">
<p style="margin: 0;">
Model pembelajaran mesin kami telah dilatih pada set data komprehensif rekod perubatan dan telah divalidasi dengan ketat untuk ketepatan. Ia menggunakan algoritma XGBoost, yang terkenal dengan prestasi tinggi dalam tugas peramalan perubatan.
</p>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- INPUT FORM ---
with st.form("prediction_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Your Health Information" if lang == "English" else "Maklumat Kesihatan Anda")

    # Group related inputs together with sections
    st.markdown("##### Basic Information")
    cols = st.columns(2)
    with cols[0]:
        inputs = {}
        inputs['Pregnancies'] = st.slider(
            input_labels[lang]["Pregnancies"],
            0, 17, 1,
            help="Number of times pregnant" if lang == "English" else "Bilangan kali mengandung"
        )
        inputs['Age'] = st.slider(
            input_labels[lang]["Age"],
            21, 81, 30,
            help="Age in years" if lang == "English" else "Umur dalam tahun"
        )

    with cols[1]:
        inputs['BMI'] = st.slider(
            input_labels[lang]["BMI"],
            0.0, 67.1, 25.0,
            step=0.1,
            help="Body Mass Index (weight in kg / height in m²)" if lang == "English" else "Indeks Jisim Badan (berat dalam kg / tinggi dalam m²)"
        )
        inputs['DiabetesPedigreeFunction'] = st.slider(
            input_labels[lang]["DiabetesPedigreeFunction"],
            0.08, 2.42, 0.47,
            step=0.01,
            help="Diabetes pedigree function (genetic influence)" if lang == "English" else "Fungsi salasilah diabetes (pengaruh genetik)"
        )

    st.markdown("##### Medical Measurements")
    cols = st.columns(2)
    with cols[0]:
        inputs['Glucose'] = st.slider(
            input_labels[lang]["Glucose"],
            0, 200, 100,
            help="Plasma glucose concentration (mg/dL)" if lang == "English" else "Konsentrasi glukosa plasma (mg/dL)"
        )
        inputs['BloodPressure'] = st.slider(
            input_labels[lang]["BloodPressure"],
            0, 122, 70,
            help="Diastolic blood pressure (mmHg)" if lang == "English" else "Tekanan darah diastolik (mmHg)"
        )

    with cols[1]:
        inputs['Insulin'] = st.slider(
            input_labels[lang]["Insulin"],
            0, 846, 79,
            help="2-Hour serum insulin (μU/mL)" if lang == "English" else "Insulin serum 2 jam (μU/mL)"
        )
        inputs['SkinThickness'] = st.slider(
            input_labels[lang]["SkinThickness"],
            0, 99, 20,
            help="Triceps skin fold thickness (mm)" if lang == "English" else "Ketebalan lipatan kulit trisep (mm)"
        )

    predict_text = {"English": "Predict Diabetes Risk", "Malay": "Ramal Risiko Diabetes"}
    submitted = st.form_submit_button(
        predict_text[lang],
        type="primary",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

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

        # Display results in a nicely styled card
        st.markdown('<div class="risk-container">', unsafe_allow_html=True)
        st.subheader("Prediction Results" if lang == "English" else "Keputusan Ramalan")

        # Risk visualization with improved styling
        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        # Set colors based on risk level
        if risk > 70:
            bar_color = '#FF4757'  # Red for high risk
        elif risk > 30:
            bar_color = '#FFCC00'  # Orange for medium risk
        else:
            bar_color = '#2ED573'  # Green for low risk

        # Create bar with improved styling
        bars = ax.barh(['Your Risk Level'], [risk], color=bar_color,
                      height=0.6, left=0, align='center')

        # Style the bar
        for bar in bars:
            bar.set_edgecolor('white')
            bar.set_linewidth(1)

        # Custom grid lines
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E0E0E0', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)

        # Remove spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Custom x-axis
        ax.set_xlabel('Risk Percentage', fontfamily='sans-serif', fontsize=12, labelpad=10)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(pad=5, color='#E0E0E0', width=0.5)
        ax.set_xticks([0, 25, 50, 75, 100])

        # Remove y-axis ticks
        ax.set_yticks([])

        # Determine text color based on bar color
        text_color = 'white' if risk > 50 else 'black'

        # Add percentage text with shadow for better visibility
        ax.text(risk/2, 0, f'{risk:.1f}%',
                ha='center', va='center',
                color=text_color,
                fontfamily='sans-serif',
                fontweight='bold',
                fontsize=14,
                bbox=dict(facecolor='rgba(255,255,255,0.7)' if risk <= 50 else 'rgba(0,0,0,0.3)',
                          edgecolor='none', boxstyle='round,pad=0.5'))

        # Add risk level indicators
        risk_levels = [
            (30, '#2ED573', 'Low'),
            (70, '#FFCC00', 'Medium'),
            (100, '#FF4757', 'High')
        ]

        for level, color, label in risk_levels:
            ax.axvline(x=level, color=color, linestyle='--', alpha=0.3, linewidth=1)
            if level == 30:
                ha = 'right'
                position = level - 2
            elif level == 100:
                ha = 'left'
                position = level - 2
            else:
                ha = 'center'
                position = level
            ax.text(position, 0.1, label,
                    ha=ha, va='center',
                    color=color,
                    fontfamily='sans-serif',
                    fontsize=10,
                    alpha=0.7)

        st.pyplot(fig, transparent=True)

        # Risk interpretation with improved styling
        if risk < 30:
            message = "🟢 Low Risk: Maintain healthy diet and exercise"
            malay_message = "🟢 Risiko Rendah: Kekalkan diet sihat dan bersenam"
            color = "#2ED573"  # Green
            icon = "✅"
        elif risk < 70:
            message = "🟠 Medium Risk: Consider lifestyle changes and monitor glucose levels"
            malay_message = "🟠 Risiko Sederhana: Pertimbangkan perubahan gaya hidup dan pantau tahap glukosa"
            color = "#FFCC00"  # Orange
            icon = "⚠️"
        else:
            message = "🔴 High Risk: Consult doctor immediately for evaluation"
            malay_message = "🔴 Risiko Tinggi: Berjumpa doktor segera untuk penilaian"
            color = "#FF4757"  # Red
            icon = "❌"

        st.markdown(f"""
        <div style="background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
                    border-left: 5px solid {color};
                    padding: 15px;
                    border-radius: 0 8px 8px 0;
                    margin-top: 1rem;
                    display: flex;
                    align-items: center;">
            <div style="font-size: 2rem; margin-right: 15px;">{icon}</div>
            <div>
                <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: {color}">
                    {message if lang == 'English' else malay_message}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Prevention tips with improved layout
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prevention & Management Tips" if lang == "English" else "Tip Pencegahan & Pengurusan")

        tips_data = {
            "English": [
                {"icon": "🍏", "title": "Healthy Diet", "text": "Maintain a balanced diet rich in fruits, vegetables, and whole grains while limiting processed foods and sugars."},
                {"icon": "🏃‍♂️", "title": "Regular Exercise", "text": "Aim for at least 150 minutes of moderate activity per week to help manage weight and blood sugar levels."},
                {"icon": "🩸", "title": "Monitor Blood Sugar", "text": "Regularly check your blood glucose levels, especially if you're at higher risk."},
                {"icon": "🚭", "title": "Avoid Smoking", "text": "Smoking increases diabetes risk and complicates management of the condition."},
                {"icon": "😴", "title": "Quality Sleep", "text": "Get 7-9 hours of quality sleep per night to help regulate hormones that affect blood sugar."},
                {"icon": "🥦", "title": "Fiber-Rich Foods", "text": "Increase intake of vegetables, whole grains, and legumes to improve blood sugar control."},
                {"icon": "💧", "title": "Stay Hydrated", "text": "Drink plenty of water and limit sugary beverages to help maintain healthy blood sugar levels."}
            ],
            "Malay": [
                {"icon": "🍏", "title": "Diet Sihat", "text": "Kekalkan diet seimbang kaya dengan buah-buahan, sayur-sayuran, dan bijirin penuh sambil menghadkan makanan diproses dan gula."},
                {"icon": "🏃‍♂️", "title": "Senaman Berkala", "text": "Sasarkan sekurang-kurangnya 150 minit aktiviti sederhana seminggu untuk membantu mengurus berat badan dan tahap gula darah."},
                {"icon": "🩸", "title": "Pantau Gula Darah", "text": "Periksa tahap glukosa darah anda secara berkala, terutama jika anda berisiko tinggi."},
                {"icon": "🚭", "title": "Elak Merokok", "text": "Merokok meningkatkan risiko diabetes dan merumitkan pengurusan keadaan tersebut."},
                {"icon": "😴", "title": "Tidur Berkualiti", "text": "Dapatkan 7-9 jam tidur berkualiti setiap malam untuk membantu mengawal hormon yang mempengaruhi gula darah."},
                {"icon": "🥦", "title": "Makanan Kaya Serat", "text": "Tingkatkan pengambilan sayur-sayuran, bijirin penuh, dan kekacang untuk mengawal tahap gula darah."},
                {"icon": "💧", "title": "Kekal Terhidrat", "text": "Minum banyak air dan hadkan minuman bergula untuk membantu mengekalkan tahap gula darah yang sihat."}
            ]
        }

        # Display tips in a responsive grid
        st.markdown('<div class="tips-grid">', unsafe_allow_html=True)

        for tip in tips_data[lang]:
            st.markdown(f"""
            <div class="tip-card">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div class="tip-icon">{tip['icon']}</div>
                    <h4 class="tip-title">{tip['title']}</h4>
                </div>
                <p class="tip-text">{tip['text']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- SIDEBAR FEATURES ---
with st.sidebar:
    st.markdown('<div class="card" style="margin-bottom: 20px;">', unsafe_allow_html=True)
    st.subheader("Model Performance")
    st.progress(MODEL_PERFORMANCE['Accuracy'])

    # Display metrics in a clean layout
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{MODEL_PERFORMANCE['Accuracy']*100:.1f}%")
        st.metric("F1 Score", f"{MODEL_PERFORMANCE['F1 Score']:.3f}")
    with col2:
        st.metric("ROC AUC", f"{MODEL_PERFORMANCE['ROC AUC']:.3f}")
        st.metric("Recall", f"{MODEL_PERFORMANCE['Recall']*100:.1f}%")

    st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Details")

    if st.checkbox("Show Feature Importance" if lang == "English" else "Tunjukkan Kepentingan Ciri"):
        try:
            st.image(FEATURE_IMG_PATH, use_column_width=True)
            st.caption("Feature importance in predicting diabetes risk" if lang == "English" else
                      "Kepentingan ciri dalam meramalkan risiko diabetes")
        except Exception as e:
            st.warning(f"Could not load feature importance image: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

    # About section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("About" if lang == "English" else "Mengenai")
    st.markdown("""
    <div style="margin-bottom: 1rem;">
    <p style="margin-bottom: 0.5rem;">This app predicts diabetes risk using machine learning.</p>
    <p style="margin-bottom: 0.5rem;">It analyzes health parameters to assess your risk level.</p>
    </div>
    """ if lang == "English" else """
    <div style="margin-bottom: 1rem;">
    <p style="margin-bottom: 0.5rem;">Aplikasi ini meramalkan risiko diabetes menggunakan pembelajaran mesin.</p>
    <p style="margin-bottom: 0.5rem;">Ia menganalisis parameter kesihatan untuk menilai tahap risiko anda.</p>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer with improved styling
    st.markdown("""
    <div style="background-color: rgba(255, 204, 0, 0.1);
                border-left: 4px solid #FFCC00;
                padding: 12px;
                border-radius: 0 8px 8px 0;
                margin-top: 1rem;">
        <p style="margin: 0; font-size: 0.9rem; color: #D35400;">
        <strong>Disclaimer:</strong> This tool provides risk estimates only. It is not a substitute for professional medical advice. Always consult a healthcare provider for medical concerns.
        </p>
    </div>
    """ if lang == "English" else """
    <div style="background-color: rgba(255, 204, 0, 0.1);
                border-left: 4px solid #FFCC00;
                padding: 12px;
                border-radius: 0 8px 8px 0;
                margin-top: 1rem;">
        <p style="margin: 0; font-size: 0.9rem; color: #D35400;">
        <strong>Penafian:</strong> Alat ini memberikan anggaran risiko sahaja. Ia bukan pengganti nasihat perubatan profesional. Sentiasa berjumpa pembekal penjagaan kesihatan untuk masalah perubatan.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.divider()
st.markdown('<div style="text-align: center; margin-top: 2rem; color: var(--dark-gray); font-size: 0.9rem;">', unsafe_allow_html=True)
st.caption("""
Developed for BIT4333 Introduction to Machine Learning | © 2023 Diabetes Risk Predictor
""")
st.markdown('</div>', unsafe_allow_html=True)

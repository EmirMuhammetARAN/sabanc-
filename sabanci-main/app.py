import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import warnings
import google.generativeai as genai
import time
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings('ignore')

# Sayfa Ayarlari
st.set_page_config(page_title="Anti-Covid AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #1e1e2f;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4ECDC4;
    }
    .metric-label {
        font-size: 1rem;
        color: #ccc;
    }
    .report-box {
        background-color: #2b2b3c;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #FF6B6B;
        font-size: 1.1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND MANTIGI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

WEIGHT_BLOOD = 0.65
WEIGHT_MRI = 0.35

GENE_NAMES = {
    'ENSG00000171517': 'LPAR3', 'ENSG00000147570': 'DNAJC5B',
    'ENSG00000172987': 'HPSE2', 'ENSG00000258810': 'LINC03058',
    'ENSG00000275243': 'TRBV16 (T-Cell)', 'ENSG00000227036': 'LINC00511',
    'ENSG00000147206': 'NXF3'
}

@st.cache_resource
def load_models():
    try:
        models = {
            'mri_scaler': joblib.load(os.path.join(MODELS_DIR, 'mri_scaler.pkl')),
            'mri_selector': joblib.load(os.path.join(MODELS_DIR, 'mri_selector.pkl')),
            'mri_model': joblib.load(os.path.join(MODELS_DIR, 'mri_model.pkl')),
            'mri_features': joblib.load(os.path.join(MODELS_DIR, 'mri_features.pkl')),
            'blood_scaler': joblib.load(os.path.join(MODELS_DIR, 'blood_scaler.pkl')),
            'blood_selector': joblib.load(os.path.join(MODELS_DIR, 'blood_selector.pkl')),
            'blood_model': joblib.load(os.path.join(MODELS_DIR, 'blood_model.pkl')),
            'blood_features': joblib.load(os.path.join(MODELS_DIR, 'blood_features.pkl'))
        }
        return models
    except Exception as e:
        return None

models = load_models()

@st.cache_data
def get_patients():
    df_mri = pd.read_excel(os.path.join(BASE_DIR, 'data', 'mri', 'Dryad_final.xlsx'), sheet_name='Combined')
    df_blood_X = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_X.csv'))
    df_blood_y = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_y.csv'))
    return df_mri, df_blood_X, df_blood_y

def run_prediction(patient_mri, patient_blood):
    # MRI
    m_scaled = models['mri_scaler'].transform(patient_mri)
    m_sel = models['mri_selector'].transform(m_scaled)
    m_dec = models['mri_model'].decision_function(m_sel)[0]
    m_prob = 1 / (1 + np.exp(-m_dec))
    m_contrib = m_sel[0] * models['mri_model'].coef_[0]
    m_names = models['mri_features'][models['mri_selector'].get_support()]
    m_top = [m_names[i] for i in np.argsort(np.abs(m_contrib))[-3:][::-1]]

    # BLOOD
    b_scaled = models['blood_scaler'].transform(patient_blood)
    b_sel = models['blood_selector'].transform(b_scaled)
    b_dec = models['blood_model'].decision_function(b_sel)[0]
    b_prob = 1 / (1 + np.exp(-b_dec))
    b_contrib = b_sel[0] * models['blood_model'].coef_[0]
    b_names = models['blood_features'][models['blood_selector'].get_support()]
    b_top = [b_names[i] for i in np.argsort(np.abs(b_contrib))[-3:][::-1]]

    final = (m_prob * WEIGHT_MRI) + (b_prob * WEIGHT_BLOOD)
    return m_prob, m_top, b_prob, b_top, final

def get_agent_report(m_prob, b_prob, final, m_top, b_top, api_key):
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel('gemini-flash-latest')
    
    trans_b = [GENE_NAMES.get(g, g) for g in b_top]
    
    prompt = f"""
    Sen uzman bir nörolog ve genetik uzmanısın. 'Anti-Covid' yapay zeka sisteminin bir asistanısın.
    Bir hastanın Long COVID (Post-COVID Sendromu) testi sonuçlandı.
    
    - Beyin MR Modeli: %{m_prob*100:.1f}
    - Kan RNA-Seq Modeli: %{b_prob*100:.1f}
    - Meta-Model Skoru: %{final*100:.1f}
    
    HASTAYA ÖZEL ANORMALLİKLER:
    - Kan: Şu genlerde anormallik var: {', '.join(trans_b)}
    - MR: Beynin şu bölgelerinde anormallik var: {', '.join(m_top)}
    
    Görev: Yukarıdaki verilere dayanarak, hastayla ilgilenen hekime yönelik kısa, profesyonel bir tıbbi rapor yaz. 
    Hastada bozulan genlerin ve beyin bölgelerinin ne anlama geldiğini açıkla, ve klinik tavsiye ver.
    """
    try:
        resp = llm.generate_content(prompt)
        # Turkce karakterleri temizle (terminalde patlamamasi icin degil, UI'da temiz dursun diye)
        return resp.text
    except Exception as e:
        return f"API Hatası: {e}"

# --- UI RENDER ---
st.markdown('<p class="main-title">Anti-Covid AI 🧠</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Multi-Modal Agentic Diagnostic System for Long COVID (PASC)</p>', unsafe_allow_html=True)

if models is None:
    st.error("Modeller yüklenemedi. Lütfen önce mri_pipeline ve blood_pipeline'ı çalıştırarak modelleri oluşturun.")
    st.stop()

df_mri, df_blood_X, df_blood_y = get_patients()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2042/2042250.png", width=100)
    st.header("Sistem Ayarları")
    api_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    
    st.markdown("---")
    st.subheader("Hasta Seçimi")
    patient_type = st.radio("Hasta Tipi:", ["Diseased Patient (Long COVID)", "Healthy Patient (Control)"])
    is_pasc_selection = True if "Diseased" in patient_type else False

    if "current_patient" not in st.session_state:
        st.session_state.current_patient = (0, 0, True)

    if st.button("🔄 Rastgele Hasta Getir", use_container_width=True) or st.session_state.current_patient[2] != is_pasc_selection:
        m_len = len(df_mri[df_mri['group'] == 'PCC'] if is_pasc_selection else df_mri[df_mri['group'] != 'PCC'])
        b_len = len(df_blood_X[df_blood_y['target'] == (1 if is_pasc_selection else 0)])
        st.session_state.current_patient = (np.random.randint(0, m_len), np.random.randint(0, b_len), is_pasc_selection)

# Rastgele Hasta Sec
is_pasc = st.session_state.current_patient[2]
mri_pool = df_mri[df_mri['group'] == 'PCC'] if is_pasc else df_mri[df_mri['group'] != 'PCC']
blood_pool = df_blood_X[df_blood_y['target'] == (1 if is_pasc else 0)]

# Pick random index
rand_m, rand_b, _ = st.session_state.current_patient

p_mri = mri_pool[models['mri_features']].iloc[rand_m:rand_m+1].fillna(0) # Eksik verileri sifirla
p_blood = blood_pool.iloc[rand_b:rand_b+1]

# Tahmin
m_prob, m_top, b_prob, b_top, final = run_prediction(p_mri, p_blood)
decision = "🔴 POZİTİF (Long COVID)" if final >= 0.5 else "🟢 NEGATİF (Sağlıklı)"

# Layout
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="metric-box"><div class="metric-label">Yapısal MR Modeli (%74 ACC)</div><div class="metric-value">%{m_prob*100:.1f}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-box"><div class="metric-label">Kan Gen-İfadesi Modeli (%97 ACC)</div><div class="metric-value">%{b_prob*100:.1f}</div></div>', unsafe_allow_html=True)
with col3:
    color = "#FF6B6B" if final >= 0.5 else "#4ECDC4"
    st.markdown(f'<div class="metric-box" style="border: 2px solid {color};"><div class="metric-label">Ensemble Meta-Model</div><div class="metric-value" style="color: {color};">%{final*100:.1f}</div></div>', unsafe_allow_html=True)

st.markdown("### 📊 Tespit Edilen Anormallikler")
st.warning(f"**MR Bulguları:** {', '.join(m_top)}")
st.error(f"**Kan (Gen) Bulguları:** {', '.join([GENE_NAMES.get(g, g) for g in b_top])}")

st.markdown("### 🤖 Agentic AI Klinik Raporu")
if st.button("✨ Raporu Oluştur (Gemini API)", type="primary"):
    with st.spinner("Gemini nöro-radyolojik ve genetik bulguları yorumluyor..."):
        report = get_agent_report(m_prob, b_prob, final, m_top, b_top, api_key)
        st.markdown(f'<div class="report-box">{report}</div>', unsafe_allow_html=True)

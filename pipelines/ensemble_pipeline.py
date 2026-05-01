import pandas as pd
import numpy as np
import os
import joblib
import warnings
import google.generativeai as genai

warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Gemini API Key (Kullanicidan gelen key)
GEMINI_API_KEY = "AIzaSyBca8I4FWQf9x2VnQqIbF6Y_7k3xvVJrMA"
genai.configure(api_key=GEMINI_API_KEY)
# Yeni gemini modeli
llm_model = genai.GenerativeModel('gemini-flash-latest')

print("=" * 60)
print("NeuroVeil: MULTI-MODAL META-LEARNER (Gercek Modeller)")
print("=" * 60)

WEIGHT_BLOOD = 0.65
WEIGHT_MRI = 0.35

# 1. Modelleri Yukle
print("[*] Egitilmis Modeller Yukleniyor...")
models_dir = os.path.join(BASE_DIR, 'models')

try:
    mri_scaler = joblib.load(os.path.join(models_dir, 'mri_scaler.pkl'))
    mri_selector = joblib.load(os.path.join(models_dir, 'mri_selector.pkl'))
    mri_model = joblib.load(os.path.join(models_dir, 'mri_model.pkl'))
    mri_features = joblib.load(os.path.join(models_dir, 'mri_features.pkl'))
    
    blood_scaler = joblib.load(os.path.join(models_dir, 'blood_scaler.pkl'))
    blood_selector = joblib.load(os.path.join(models_dir, 'blood_selector.pkl'))
    blood_model = joblib.load(os.path.join(models_dir, 'blood_model.pkl'))
    blood_features = joblib.load(os.path.join(models_dir, 'blood_features.pkl'))
except FileNotFoundError:
    print("HATA: Modeller bulunamadi. Once mri_pipeline.py ve blood_pipeline.py calistirilmali!")
    exit()

GENE_NAMES = {
    'ENSG00000171517': 'LPAR3',
    'ENSG00000147570': 'DNAJC5B',
    'ENSG00000172987': 'HPSE2',
    'ENSG00000258810': 'LINC03058',
    'ENSG00000275243': 'TRBV16 (T-Cell Receptor)',
    'ENSG00000227036': 'LINC00511',
    'ENSG00000147206': 'NXF3'
}

def get_real_patient_data():
    """Test setinden rastgele gercek bir hastanin verilerini secer."""
    # MR Verisi
    df_mri = pd.read_excel(os.path.join(BASE_DIR, 'data', 'mri', 'Dryad_final.xlsx'), sheet_name='Combined')
    mri_patient = df_mri[df_mri['group'] == 'PCC'][mri_features].iloc[1:2]
    # Eksik verileri doldur ki SVM 0.5 hata vermesin
    mri_patient = mri_patient.fillna(mri_patient.mean().fillna(0))
    
    # Kan Verisi
    df_blood_X = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_X.csv'))
    df_blood_y = pd.read_csv(os.path.join(BASE_DIR, 'data', 'blood', 'processed', 'blood_y.csv'))
    blood_patient = df_blood_X[df_blood_y['target'] == 1].iloc[1:2]
    
    return mri_patient, blood_patient

def run_mri_model(patient_df):
    """Gercek MR Pipeline'indan gecir."""
    X_scaled = mri_scaler.transform(patient_df)
    X_sel = mri_selector.transform(X_scaled)
    dec = mri_model.decision_function(X_sel)[0]
    prob = 1 / (1 + np.exp(-dec)) # Sigmoid
    
    # Hastaya Ozel Feature Importance
    contributions = X_sel[0] * mri_model.coef_[0]
    selected_names = mri_features[mri_selector.get_support()]
    top_idx = np.argsort(np.abs(contributions))[-3:][::-1]
    top_mri = [selected_names[i] for i in top_idx]
    
    return prob, top_mri

def run_blood_model(patient_df):
    """Gercek Kan Pipeline'indan gecir."""
    X_scaled = blood_scaler.transform(patient_df)
    X_sel = blood_selector.transform(X_scaled)
    dec = blood_model.decision_function(X_sel)[0]
    prob = 1 / (1 + np.exp(-dec)) # Sigmoid
    
    # Hastaya Ozel Feature Importance
    contributions = X_sel[0] * blood_model.coef_[0]
    selected_names = blood_features[blood_selector.get_support()]
    top_idx = np.argsort(np.abs(contributions))[-3:][::-1]
    top_blood = [selected_names[i] for i in top_idx]
    
    return prob, top_blood

def generate_agentic_report(mri_prob, blood_prob, final_score, decision, top_mri, top_blood):
    """Gemini API kullanarak gercek zamanli doktor raporu uretir."""
    print("\n[*] Agentic AI (Gemini) Doktor Raporu Hazirliyor...")
    
    translated_blood = [GENE_NAMES.get(g, g) for g in top_blood]
    
    prompt = f"""
    Sen uzman bir nörolog ve genetik uzmanısın. 'NeuroVeil' adlı yapay zeka sisteminin bir asistanısın.
    Bir hastanın Long COVID (Post-COVID Sendromu) testi sonuçlandı.
    
    Yapay Zeka Modellerinin Çıktıları:
    - Yapısal Beyin MR Modeli (Sadece anatomiye bakar): Hastanın Long COVID olma ihtimalini %{mri_prob*100:.1f} olarak hesapladı.
    - Kan RNA-Seq Modeli (Gen ifadesi ve bağışıklığa bakar): Hastanın Long COVID olma ihtimalini %{blood_prob*100:.1f} olarak hesapladı.
    - NeuroVeil Ensemble Meta-Model Sonucu: Hastanın toplam Long COVID skoru %{final_score*100:.1f}
    - Sistem Kararı: {decision}
    
    Önemli Ek Bilgiler (HASTAYA ÖZEL ANORMALLİKLER):
    - Kan analizinde, özellikle şu genlerde şiddetli anormallik/hareketlilik tespit edildi: {', '.join(translated_blood)}
    - MR analizinde, beynin özellikle şu bölgelerinde anormal yapısal sinyaller var: {', '.join(top_mri)}
    
    Görev: 
    Yukarıdaki verilere dayanarak, hastayla ilgilenen hekime yönelik kısa, profesyonel ve klinik bir dille bir teşhis/değerlendirme raporu yaz.
    Rapor en fazla 3-4 paragraf olsun. 
    İçeriğinde hastanın MR ve Kan yapay zeka skorlarını yorumla, ÖZELLİKLE HASTADA BOZULAN bu genlerin ve beyin bölgelerinin ne anlama geldiğini açıkla, ve son bir hekim tavsiyesi ver.
    """
    
    try:
        response = llm_model.generate_content(prompt)
        print("\n" + "="*50)
        print("AGENTIC AI KLINIK RAPORU")
        print("="*50)
        safe_text = response.text.replace('ı', 'i').replace('İ', 'I').replace('ş', 's').replace('Ş', 'S').replace('ğ', 'g').replace('Ğ', 'G').replace('ü', 'u').replace('Ü', 'U').replace('ö', 'o').replace('Ö', 'O').replace('ç', 'c').replace('Ç', 'C')
        print(safe_text)
        print("="*50)
    except Exception as e:
        print(f"Agentic AI Hatasi: {e}")

if __name__ == "__main__":
    print("[*] Veritabanindan gercek hasta verileri cekiliyor...")
    mri_data, blood_data = get_real_patient_data()
    
    print("[*] Hasta beyin MR verisi yapay zekada isleniyor...")
    mri_prob, top_mri = run_mri_model(mri_data)
    
    print("[*] Hasta kan gen-ifadesi verisi yapay zekada isleniyor...")
    blood_prob, top_blood = run_blood_model(blood_data)
    
    print("[*] Ensemble Late Fusion uygulaniyor...")
    final_score = (mri_prob * WEIGHT_MRI) + (blood_prob * WEIGHT_BLOOD)
    decision = "POZITIF (Long COVID Tespit Edildi)" if final_score >= 0.50 else "NEGATIF (Saglikli / Iyilesmis)"
    
    print("\n--- NEUROVEIL TESHIS PANELI ---")
    print(f" MR Skoru:   %{mri_prob*100:.1f}")
    print(f" Kan Skoru:  %{blood_prob*100:.1f}")
    print("-" * 35)
    print(f" FINAL SKOR: %{final_score*100:.1f}")
    print(f" KARAR:      {decision}")
    print(f" Ozet Bozukluklar: {', '.join(top_mri)} | {', '.join([GENE_NAMES.get(g, g) for g in top_blood])}")
    
    # Gemini'a rapor yazdir
    generate_agentic_report(mri_prob, blood_prob, final_score, decision, top_mri, top_blood)

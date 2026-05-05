"""
Anti-Covid AI — FastAPI Backend v2.1
Auth + JWT + GradientBoosting modelleri

Calistirma:
    cd sabanci-main
    uvicorn backend.main:app --reload --port 8000
"""

import os, sys, io, warnings
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import imblearn
import xgboost
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import JWTError, jwt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from database.db import (
    supabase,
    authenticate_doctor,
    create_doctor,
    get_doctor_by_id,
    get_doctor_by_email,
    upsert_patient,
    get_patient_by_id,
    search_patients,
    save_analysis,
    update_analysis_report,
    get_analyses_for_patient,
    get_analysis_by_id,
    get_doctor_dashboard_stats,
    log_action,
    hash_tc,
)

load_dotenv()
warnings.filterwarnings('ignore')

# ── Sabitler ─────────────────────────────────
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
WEIGHT_MRI    = 0.35
WEIGHT_BLOOD  = 0.65
JWT_SECRET    = os.environ.get("JWT_SECRET", "neuroveil-super-secret-key-2024")
JWT_ALGORITHM = "HS256"
JWT_EXP_HOURS = 24

ML = {}


# ══════════════════════════════════════════════
# JWT YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════

def create_token(doctor_id: str, email: str, role: str) -> str:
    payload = {
        "sub": doctor_id,
        "email": email,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXP_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Gecersiz veya suresi dolmus token.")


security = HTTPBearer()

def get_current_doctor(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Tum korunan endpoint'lere inject edilir. Token gecersizse 401 verir."""
    payload = decode_token(credentials.credentials)
    doctor = get_doctor_by_id(payload["sub"])
    if not doctor or not doctor.get("is_active"):
        raise HTTPException(status_code=401, detail="Doktor bulunamadi veya hesap pasif.")
    return doctor


# ══════════════════════════════════════════════
# MODEL YUKLE
# ══════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Anti-Covid modelleri yukleniyor...")
    try:
        # --- MRI Pipeline Yukle ---
        mri_pipe_path = os.path.join(MODELS_DIR, 'mri_champion_pipeline.pkl')
        if os.path.exists(mri_pipe_path):
            mri_pipe = joblib.load(mri_pipe_path)
            ML['mri_scaler']   = mri_pipe.named_steps['scaler']
            ML['mri_selector'] = mri_pipe.named_steps['selector']
            ML['mri_model']    = mri_pipe.named_steps['model']
            
            # Scaler'dan orijinal ozellik listesini al (reindex icin)
            if hasattr(ML['mri_scaler'], 'feature_names_in_'):
                ML['mri_features'] = ML['mri_scaler'].feature_names_in_
            else:
                ML['mri_features'] = joblib.load(os.path.join(MODELS_DIR, 'mri_features.pkl'))
            print("[OK] MRI modeli yuklendi.")
        else:
            print(f"[!] MRI pipeline bulunamadi: {mri_pipe_path}")

        # --- Blood Pipeline Yukle ---
        blood_pipe_path = os.path.join(MODELS_DIR, 'blood_champion_pipeline.pkl')
        if os.path.exists(blood_pipe_path):
            blood_pipe = joblib.load(blood_pipe_path)
            if 'var_thresh' in blood_pipe.named_steps:
                ML['blood_var_thresh'] = blood_pipe.named_steps['var_thresh']
            
            ML['blood_scaler']   = blood_pipe.named_steps['scaler']
            ML['blood_selector'] = blood_pipe.named_steps['selector']
            ML['blood_model']    = blood_pipe.named_steps['model']
            
            # İlk adimdan orijinal gen listesini al
            first_step = blood_pipe.steps[0][1]
            if hasattr(first_step, 'feature_names_in_'):
                ML['blood_features'] = first_step.feature_names_in_
            else:
                ML['blood_features'] = joblib.load(os.path.join(MODELS_DIR, 'blood_features.pkl'))
            print("[OK] Kan (Blood) modeli yuklendi.")
        else:
            print(f"[!] Blood pipeline bulunamadi: {blood_pipe_path}")

        # --- Groq Konfigurasyonu (Öncelikli) ---
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            ML['groq'] = Groq(api_key=groq_key)
            print("[OK] Groq AI hazir (Llama-3).")

        # --- Gemini Konfigurasyonu (Yedek) ---
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            ML['gemini'] = genai.GenerativeModel('gemini-2.0-flash-lite')
            print("[OK] Gemini AI hazir.")
        
        if not groq_key and not gemini_key:
            print("[!] AI Key bulunamadi, rapor uretimi otomatik moda gececek.")

        if 'mri_model' in ML and 'blood_model' in ML:
            print("[SUCCESS] Tum ML modelleri basariyla aktif edildi.", flush=True)
            
    except Exception as e:
        print(f"[ERROR] Lifespan hatasi: {e}")
    yield
    ML.clear()


app = FastAPI(
    title="Anti-Covid AI API",
    description="Long COVID cok-modal tani yardim sistemi",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # During local development allow localhost; in production prefer explicit origins
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════
# AUTH ENDPOİNT'LERİ
# ══════════════════════════════════════════════

class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    hospital: Optional[str] = None
    department: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/api/auth/register", summary="Yeni doktor kaydı")
def register(body: RegisterRequest):
    """
    Yeni doktor hesabı oluşturur.
    E-posta zaten kayıtlıysa 400 döner.
    """
    existing = get_doctor_by_email(body.email)
    if existing:
        raise HTTPException(status_code=400, detail="Bu e-posta zaten kayitli.")

    doctor = create_doctor(
        email=body.email,
        password=body.password,
        full_name=body.full_name,
        hospital=body.hospital,
        department=body.department,
    )

    token = create_token(doctor["id"], doctor["email"], doctor["role"])
    log_action(doctor_id=doctor["id"], action="LOGIN")

    return {
        "token": token,
        "doctor": {
            "id": doctor["id"],
            "email": doctor["email"],
            "full_name": doctor["full_name"],
            "role": doctor["role"],
            "hospital": doctor.get("hospital"),
        }
    }


@app.post("/api/auth/login", summary="Doktor girişi")
def login(body: LoginRequest):
    """
    Email + şifre ile giriş yapar, JWT token döner.
    Frontend bu token'ı Authorization: Bearer <token> header'ında gönderir.
    """
    doctor = authenticate_doctor(body.email, body.password)
    if not doctor:
        raise HTTPException(status_code=401, detail="Email veya sifre yanlis.")
    if not doctor.get("is_active"):
        raise HTTPException(status_code=403, detail="Hesabiniz pasif durumda.")

    token = create_token(doctor["id"], doctor["email"], doctor["role"])
    log_action(doctor_id=doctor["id"], action="LOGIN")

    return {
        "token": token,
        "doctor": {
            "id": doctor["id"],
            "email": doctor["email"],
            "full_name": doctor["full_name"],
            "role": doctor["role"],
            "hospital": doctor.get("hospital"),
        }
    }


@app.get("/api/auth/me", summary="Mevcut doktor bilgisi")
def me(current_doctor: dict = Depends(get_current_doctor)):
    """Token geçerliyse doktor profilini döner. Frontend'de 'ben kimim' kontrolü için."""
    return {
        "id": current_doctor["id"],
        "email": current_doctor["email"],
        "full_name": current_doctor["full_name"],
        "role": current_doctor["role"],
        "hospital": current_doctor.get("hospital"),
        "department": current_doctor.get("department"),
    }


# ── Health check ─────────────────────────────
@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Anti-Covid AI API",
        "version": "2.1.0",
        "models_loaded": len(ML) > 0,
    }


# ══════════════════════════════════════════════
# HASTA (korumalı)
# ══════════════════════════════════════════════

class PatientCreate(BaseModel):
    tc_no: str
    full_name: str
    birth_date: Optional[str] = None
    gender: Optional[str] = None


@app.post("/api/patients")
def create_or_update_patient(
    body: PatientCreate,
    current_doctor: dict = Depends(get_current_doctor),
):
    patient = upsert_patient(
        tc_no=body.tc_no,
        full_name=body.full_name,
        birth_date=body.birth_date,
        gender=body.gender,
    )
    log_action(doctor_id=current_doctor["id"], action="PATIENT_CREATE",
               target_id=patient["id"], target_type="patient")
    return {"success": True, "patient": patient}


@app.get("/api/patients/search")
def search_patient(q: str, current_doctor: dict = Depends(get_current_doctor)):
    if len(q) < 2:
        raise HTTPException(status_code=400, detail="En az 2 karakter girin.")
    return {"patients": search_patients(q)}


@app.get("/api/patients/{patient_id}")
def get_patient(patient_id: str, current_doctor: dict = Depends(get_current_doctor)):
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Hasta bulunamadi.")
    log_action(doctor_id=current_doctor["id"], action="PATIENT_VIEW",
               target_id=patient_id, target_type="patient")
    return {"patient": patient, "analyses": get_analyses_for_patient(patient_id)}


# ══════════════════════════════════════════════
# MODEL YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════

def get_top_features(model, selected_names, n=3):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return [str(selected_names[i]) for i in range(min(n, len(selected_names)))]
    top_idx = np.argsort(importances)[-n:][::-1]
    return [str(selected_names[i]) for i in top_idx]


def run_mri_model(patient_df: pd.DataFrame):
    X_scaled = ML['mri_scaler'].transform(patient_df)
    X_sel    = ML['mri_selector'].transform(X_scaled)
    prob     = float(ML['mri_model'].predict_proba(X_sel)[0][1])
    selected = ML['mri_features'][ML['mri_selector'].get_support()]
    return prob, get_top_features(ML['mri_model'], selected)


def run_blood_model(patient_df: pd.DataFrame):
    X = patient_df
    # Variance Threshold varsa uygula
    if 'blood_var_thresh' in ML:
        X = ML['blood_var_thresh'].transform(X)
    
    X_scaled = ML['blood_scaler'].transform(X)
    X_sel    = ML['blood_selector'].transform(X_scaled)
    prob     = float(ML['blood_model'].predict_proba(X_sel)[0][1])
    
    # Maskeleri sirayla uygulayarak secilen gen isimlerini bul
    current_features = ML['blood_features']
    if 'blood_var_thresh' in ML:
        current_features = current_features[ML['blood_var_thresh'].get_support()]
    
    selected_genes = current_features[ML['blood_selector'].get_support()]
    
    return prob, get_top_features(ML['blood_model'], selected_genes)


def generate_ai_report(mri_prob, blood_prob, final_score, decision, top_mri, top_blood) -> str:
    prompt = f"""
[SYSTEM: ANTI-COVID AI CLINICAL DIAGNOSTIC ASSISTANT]
Patient has {decision} results for Long COVID (PASC).
Ensemble Score: {final_score*100:.1f}%
MRI Confidence: {mri_prob*100:.1f}% | Blood Confidence: {blood_prob*100:.1f}%
Abnormalities — Blood genes: {', '.join(top_blood)} | MRI regions: {', '.join(top_mri)}
Write a concise, professional, clinical English report for the physician (3-4 paragraphs).
"""
    # 1. Try Groq first
    if 'groq' in ML:
        try:
            response = ML['groq'].chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[!] Groq Report Error: {str(e)[:100]}...")

    # 2. Fallback to Gemini if Groq fails
    try:
        if 'gemini' in ML:
            return ML['gemini'].generate_content(prompt).text
    except Exception as e:
        print(f"[!] AI Report Error: {e}")
    
    # --- FALLBACK REPORT (If all APIs fail) ---
    status_text = "POSITIVE" if decision == "POSITIVE" else "NEGATIVE"
    return f"""
ANTI-COVID AI - CLINICAL ANALYSIS REPORT (AUTOMATIC)
--------------------------------------------------
Analysis Date: {datetime.now().strftime('%m/%d/%Y %H:%M')}
Result: {status_text} (Score: {final_score*100:.1f}%)

MODEL OUTPUTS:
- Brain MRI Model Confidence: {mri_prob*100:.1f}%
- Blood RNA-Seq Model Confidence: {blood_prob*100:.1f}%

CRITICAL FINDINGS:
- Blood Gene Expression Abnormalities: {', '.join(top_blood) if top_blood else 'Normal'}
- MRI Volumetric Deviations: {', '.join(top_mri) if top_mri else 'Normal'}

NOTE: A detailed text report could not be generated at this time due to temporary quota limits or connectivity issues with the AI service (LLM Model). 
The data above are the raw analysis results derived directly from the ML models.
"""


# ══════════════════════════════════════════════
# ANALİZ (korumalı)
# ══════════════════════════════════════════════

@app.post("/api/predict")
async def predict(
    mri_file:   UploadFile = File(...),
    blood_file: UploadFile = File(...),
    patient_id: str        = Form(...),
    save_to_db: bool       = Form(True),
    current_doctor: dict   = Depends(get_current_doctor),
):
    if not ML:
        raise HTTPException(status_code=503, detail="Modeller yuklenmedi.")

    try:
        df_mri       = pd.read_excel(io.BytesIO(await mri_file.read()), sheet_name='Combined')
        mri_features = ML['mri_features']
        patient_mri  = df_mri.iloc[0:1].reindex(columns=mri_features, fill_value=0).fillna(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"MRI dosyasi hatasi: {str(e)}")

    try:
        df_blood       = pd.read_csv(io.BytesIO(await blood_file.read()))
        df_blood       = df_blood.apply(pd.to_numeric, errors='coerce').fillna(0)
        blood_features = ML['blood_features']
        patient_blood  = df_blood.iloc[0:1].reindex(columns=blood_features, fill_value=0).fillna(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Kan dosyasi hatasi: {str(e)}")

    try:
        mri_prob,   top_mri   = run_mri_model(patient_mri)
        blood_prob, top_blood = run_blood_model(patient_blood)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model hatasi: {str(e)}")

    final_score     = (mri_prob * WEIGHT_MRI) + (blood_prob * WEIGHT_BLOOD)
    decision        = "POSITIVE" if final_score >= 0.50 else "NEGATIVE"
    top_blood_named = top_blood

    result = {
        "mri_score":        round(mri_prob * 100, 1),
        "blood_score":      round(blood_prob * 100, 1),
        "ensemble_score":   round(final_score * 100, 1),
        "decision":         decision,
        "top_mri_features": top_mri,
        "top_blood_genes":  top_blood_named,
        "analysis_id":      None,
    }

    if save_to_db:
        try:
            analysis = save_analysis(
                patient_id=patient_id,
                doctor_id=current_doctor["id"],
                mri_score=mri_prob,
                blood_score=blood_prob,
                ensemble_score=final_score,
                decision=decision,
                top_mri_features=top_mri,
                top_blood_genes=top_blood_named,
                raw_mri_input=patient_mri.to_dict(orient='records')[0],
                raw_blood_input=patient_blood.to_dict(orient='records')[0],
            )
            result["analysis_id"] = analysis["id"] if analysis else None
            if not analysis:
                # Surface DB failure so frontend can handle it instead of silently using temp route
                raise HTTPException(status_code=500, detail="DB kaydi basarisiz.")
            log_action(doctor_id=current_doctor["id"], action="ANALYSIS_RUN",
                       target_id=analysis["id"], target_type="analysis")
        except Exception as e:
            print(f"[!] DB kayit hatasi: {e}")
            raise

    return result


# ── Rapor ────────────────────────────────────
class ReportRequest(BaseModel):
    analysis_id: str


@app.post("/api/report")
def generate_report(
    body: ReportRequest,
    current_doctor: dict = Depends(get_current_doctor),
):
    analysis = get_analysis_by_id(body.analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analiz bulunamadi.")

    report = generate_ai_report(
        mri_prob    = analysis["mri_score"] / 100,
        blood_prob  = analysis["blood_score"] / 100,
        final_score = analysis["ensemble_score"] / 100,
        decision    = analysis["decision"],
        top_mri     = analysis["top_mri_features"] or [],
        top_blood   = analysis["top_blood_genes"] or [],
    )
    update_analysis_report(body.analysis_id, report)
    log_action(doctor_id=current_doctor["id"], action="REPORT_GENERATE",
               target_id=body.analysis_id, target_type="analysis")

    return {"report": report, "analysis_id": body.analysis_id}


# ── Dashboard ─────────────────────────────────
@app.get("/api/dashboard")
def doctor_dashboard(current_doctor: dict = Depends(get_current_doctor)):
    return get_doctor_dashboard_stats(current_doctor["id"])


@app.get("/api/analyses/{analysis_id}")
def get_analysis(
    analysis_id: str,
    current_doctor: dict = Depends(get_current_doctor),
):
    analysis = get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analiz bulunamadi.")
    return analysis


@app.get("/api/patients/{patient_id}/analyses")
def get_patient_analyses(
    patient_id: str,
    current_doctor: dict = Depends(get_current_doctor),
):
    return {"analyses": get_analyses_for_patient(patient_id)}

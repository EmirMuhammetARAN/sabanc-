import os
import hashlib
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from passlib.context import CryptContext

load_dotenv()

SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]
# service_role key — RLS'yi atlar, sadece backend'de kullanılır
SUPABASE_SERVICE_KEY: str = os.environ.get("SUPABASE_SERVICE_KEY", SUPABASE_KEY)

# Okuma için anon client, yazma için service client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Yardımcı ─────────────────────────────────
def hash_tc(tc_no: str) -> str:
    return hashlib.sha256(tc_no.strip().encode()).hexdigest()

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ══════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════

def get_doctor_by_email(email: str) -> Optional[dict]:
    res = supabase_admin.table("doctors").select("*").eq("email", email).execute()
    return res.data[0] if res.data else None

def get_doctor_by_id(doctor_id: str) -> Optional[dict]:
    res = supabase_admin.table("doctors").select("*").eq("id", doctor_id).execute()
    return res.data[0] if res.data else None

def create_doctor(
    email: str,
    password: str,
    full_name: str,
    hospital: str = None,
    department: str = None,
    role: str = "doctor",
) -> dict:
    # Admin client ile yaz — RLS atlanır
    res = supabase_admin.table("doctors").insert({
        "email": email,
        "password_hash": hash_password(password),
        "full_name": full_name,
        "hospital": hospital,
        "department": department,
        "role": role,
    }).execute()
    return res.data[0]

def authenticate_doctor(email: str, password: str) -> Optional[dict]:
    doctor = get_doctor_by_email(email)
    if not doctor:
        return None
    if not doctor.get("password_hash"):
        return None
    if not verify_password(password, doctor["password_hash"]):
        return None
    return doctor


# ══════════════════════════════════════════════
# PATIENTS
# ══════════════════════════════════════════════

def get_patient_by_tc(tc_no: str) -> Optional[dict]:
    tc_h = hash_tc(tc_no)
    res = supabase_admin.table("patients").select("*").eq("tc_hash", tc_h).execute()
    return res.data[0] if res.data else None

def get_patient_by_id(patient_id: str) -> Optional[dict]:
    res = supabase_admin.table("patients").select("*").eq("id", patient_id).execute()
    return res.data[0] if res.data else None

def search_patients(name_query: str, limit: int = 20) -> list:
    res = (
        supabase_admin.table("patients")
        .select("id, full_name, birth_date, gender, enabiz_synced_at")
        .ilike("full_name", f"%{name_query}%")
        .limit(limit)
        .execute()
    )
    return res.data

def upsert_patient(
    tc_no: str,
    full_name: str,
    birth_date: str = None,
    gender: str = None,
    covid_history: dict = None,
    anamnesis: dict = None,
) -> dict:
    tc_h = hash_tc(tc_no)
    payload = {
        "tc_hash": tc_h,
        "full_name": full_name,
        "birth_date": birth_date,
        "gender": gender,
        "covid_history": covid_history or {},
        "anamnesis": anamnesis or {},
        "enabiz_synced_at": datetime.utcnow().isoformat(),
    }
    res = supabase_admin.table("patients").upsert(payload, on_conflict="tc_hash").execute()
    return res.data[0]


# ══════════════════════════════════════════════
# ANALYSES
# ══════════════════════════════════════════════

def save_analysis(
    patient_id: str,
    doctor_id: str,
    mri_score: float,
    blood_score: float,
    ensemble_score: float,
    decision: str,
    top_mri_features: list,
    top_blood_genes: list,
    gemini_report: str = None,
    raw_mri_input: dict = None,
    raw_blood_input: dict = None,
    mri_file_path: str = None,
    blood_file_path: str = None,
    model_version: str = "v2.0",
) -> dict:
    res = supabase_admin.table("analyses").insert({
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "mri_score": round(mri_score * 100, 2),
        "blood_score": round(blood_score * 100, 2),
        "ensemble_score": round(ensemble_score * 100, 2),
        "decision": decision.upper(),
        "top_mri_features": top_mri_features,
        "top_blood_genes": top_blood_genes,
        "gemini_report": gemini_report,
        "raw_mri_input": raw_mri_input or {},
        "raw_blood_input": raw_blood_input or {},
        "mri_file_path": mri_file_path,
        "blood_file_path": blood_file_path,
        "model_version": model_version,
    }).execute()
    inserted = res.data[0] if res.data else None
    try:
        print(f"[DB] Saved analysis id={inserted.get('id') if inserted else None}")
    except Exception:
        print(f"[DB] Saved analysis, response: {res}")
    return inserted

def update_analysis_report(analysis_id: str, gemini_report: str) -> dict:
    res = (
        supabase_admin.table("analyses")
        .update({"gemini_report": gemini_report})
        .eq("id", analysis_id)
        .execute()
    )
    return res.data[0]

def get_analyses_for_patient(patient_id: str, limit: int = 10) -> list:
    res = (
        supabase_admin.table("analyses")
        .select("id, ensemble_score, decision, created_at, doctor_id, top_blood_genes, top_mri_features")
        .eq("patient_id", patient_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data

def get_analysis_by_id(analysis_id: str) -> Optional[dict]:
    try:
        res = supabase_admin.table("analyses").select("*").eq("id", analysis_id).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        # Usually occurs when `analysis_id` isn't a valid UUID for the DB column.
        print(f"[!] get_analysis_by_id error: {e}")
        return None

def get_doctor_dashboard_stats(doctor_id: str) -> dict:
    res = (
        supabase_admin.table("analyses")
        .select("id, patient_id, decision, ensemble_score, created_at, top_blood_genes, top_mri_features")
        .eq("doctor_id", doctor_id)
        .order("created_at", desc=True)
        .limit(100)
        .execute()
    )
    rows = res.data
    total = len(rows)
    positive = sum(1 for r in rows if r["decision"] == "POSITIVE")
    avg_score = sum(r["ensemble_score"] for r in rows) / total if total else 0
    return {
        "total_analyses": total,
        "positive_count": positive,
        "negative_count": total - positive,
        "positive_rate": round(positive / total * 100, 1) if total else 0,
        "avg_ensemble_score": round(avg_score, 1),
        "recent_analyses": rows[:5],
    }


# ══════════════════════════════════════════════
# AUDIT LOGS
# ══════════════════════════════════════════════

def log_action(
    doctor_id: str,
    action: str,
    target_id: str = None,
    target_type: str = None,
    ip_address: str = None,
    user_agent: str = None,
    meta: dict = None,
) -> None:
    try:
        supabase_admin.table("audit_logs").insert({
            "doctor_id": doctor_id,
            "action": action,
            "target_id": target_id,
            "target_type": target_type,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "meta": meta or {},
        }).execute()
    except Exception as e:
        print(f"[!] Audit log hatasi: {e}")

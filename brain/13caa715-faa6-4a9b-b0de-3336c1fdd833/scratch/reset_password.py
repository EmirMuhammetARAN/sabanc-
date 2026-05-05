import os
from supabase import create_client, Client
from dotenv import load_dotenv
from passlib.context import CryptContext

load_dotenv(os.path.join(os.path.dirname(__file__), "../../../sabanci-main/.env"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def reset_all_passwords():
    doctors = ["berke@anti-covid.ai", "test.doktor@anti-covid.ai"]
    new_password = "test1234"
    password_hash = hash_password(new_password)
    
    for email in doctors:
        print(f"Resetting password for {email} to {new_password}...")
        res = supabase.table("doctors").update({"password_hash": password_hash}).eq("email", email).execute()
        if res.data:
            print(f"[OK] Password reset successful for {email}")
        else:
            print(f"[!] Could not reset password for {email}")

if __name__ == "__main__":
    reset_all_passwords()

import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../../sabanci-main/.env"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def update_doctors():
    doctors = [
        {"old": "berke@neuroveil.ai", "new": "berke@anti-covid.ai"},
        {"old": "test.doktor@neuroveil.ai", "new": "test.doktor@anti-covid.ai"}
    ]
    
    for d in doctors:
        print(f"Updating {d['old']} -> {d['new']}...")
        res = supabase.table("doctors").update({"email": d["new"]}).eq("email", d["old"]).execute()
        if res.data:
            print(f"[OK] Successfully updated {d['new']}")
        else:
            print(f"[!] Could not update {d['old']} (maybe it doesn't exist or is already updated)")

if __name__ == "__main__":
    update_doctors()

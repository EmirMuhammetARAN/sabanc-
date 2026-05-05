import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Point to the .env file in sabanci-main
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../sabanci-main/.env"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    # Fallback to local .env if above fails
    load_dotenv()
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print(f"Error: SUPABASE_URL or SUPABASE_SERVICE_KEY not found. CWD: {os.getcwd()}")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def list_doctors():
    res = supabase.table("doctors").select("*").execute()
    print(f"Total doctors: {len(res.data)}")
    for d in res.data:
        print(f"- ID: {d['id']}, Email: {d['email']}, Active: {d.get('is_active')}")

if __name__ == "__main__":
    list_doctors()

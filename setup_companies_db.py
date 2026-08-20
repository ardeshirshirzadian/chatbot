"""One-off script: creates the `companies` table in Postgres if it doesn't exist yet.

This table mirrors exhibitor/company data pushed from the admin panel via
POST /admin/companies/sync — see main.py.
"""
import psycopg2

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "user": "chatbot",
    "password": "Xk7#mQ2vN9pL$wR4tZ8j",
    "dbname": "chatbot_faq",
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY,
    rasayesh_event_id INTEGER,
    event_id INTEGER NOT NULL DEFAULT 1,
    brand_name_fa TEXT,
    brand_name_en TEXT,
    legal_name_fa TEXT,
    legal_name_en TEXT,
    logo JSONB,
    website TEXT,
    description_fa TEXT,
    description_en TEXT,
    slug TEXT,
    phones JSONB,
    emails JSONB,
    address_fa TEXT,
    address_en TEXT,
    industry_id INTEGER,
    hall_name TEXT,
    booth_no TEXT,
    is_sponsor BOOLEAN DEFAULT false,
    sponsor_level TEXT,
    booth_uuid UUID,
    booth_xp INTEGER,
    is_manual BOOLEAN DEFAULT false,
    linked_mission_id INTEGER,
    linked_badge_id INTEGER,
    repeatable_scan BOOLEAN DEFAULT false,
    repeatable_scan_hours INTEGER,
    repeatable_start_hour INTEGER,
    synced_at TIMESTAMP DEFAULT NOW()
);
"""

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
        print("✅ Table `companies` created (or already existed).")
    finally:
        conn.close()

if __name__ == "__main__":
    main()

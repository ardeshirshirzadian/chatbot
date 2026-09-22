"""One-off script: creates the `panels` table in Postgres if it doesn't exist yet.

This table mirrors panel/workshop data pushed from the admin panel via
POST /admin/panels/sync -- see main.py. Mirrors setup_companies_db.py.
"""
import psycopg2

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5434,
    "user": "chatbot",
    "password": "Xk7#mQ2vN9pL$wR4tZ8j",
    "dbname": "chatbot_faq",
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS panels (
    id INTEGER PRIMARY KEY,
    rasayesh_event_id INTEGER,
    event_id INTEGER NOT NULL DEFAULT 1,
    title_fa TEXT,
    title_en TEXT,
    description_fa TEXT,
    description_en TEXT,
    hall_fa TEXT,
    hall_en TEXT,
    starts_at TIMESTAMP,
    ends_at TIMESTAMP,
    capacity INTEGER,
    kind TEXT,
    thumbnail JSONB,
    speakers JSONB,
    synced_at TIMESTAMP DEFAULT NOW()
);
"""

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
        print("✅ Table `panels` created (or already existed).")
    finally:
        conn.close()

if __name__ == "__main__":
    main()

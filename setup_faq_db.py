"""One-off script: creates the `faq` table in Postgres if it doesn't exist yet."""
import psycopg2

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "user": "chatbot",
    "password": "Xk7#mQ2vN9pL$wR4tZ8j",
    "dbname": "chatbot_faq",
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS faq (
    id TEXT PRIMARY KEY,
    category TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
"""

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
        print("✅ Table `faq` created (or already existed).")
    finally:
        conn.close()

if __name__ == "__main__":
    main()

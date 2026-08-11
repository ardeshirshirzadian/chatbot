"""One-off script: creates the `bot_settings` table in Postgres if it doesn't
exist yet, and seeds the initial `fallback_message` row.

Generic key-value table for bot-wide settings (starting with the fallback
message, shaped to hold more settings later — one row per key).
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
CREATE TABLE IF NOT EXISTS bot_settings (
    key TEXT PRIMARY KEY,
    value_fa TEXT,
    value_en TEXT,
    updated_at TIMESTAMP DEFAULT NOW()
);
"""

SEED_FALLBACK_SQL = """
INSERT INTO bot_settings (key, value_fa, value_en)
VALUES (%s, %s, %s)
ON CONFLICT (key) DO NOTHING;
"""

FALLBACK_FA = "این سؤال خارج از حوزه نمایشگاه ایران‌فارما است یا اطلاعات آن در پایگاه دانش ثبت نشده است."
FALLBACK_EN = "This question is outside the scope of the IranPharma exhibition, or the information has not been recorded in the knowledge base."


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                cur.execute(SEED_FALLBACK_SQL, ("fallback_message", FALLBACK_FA, FALLBACK_EN))
        print("✅ Table `bot_settings` created (or already existed).")
        print("✅ Seeded `fallback_message` row (or already present).")
    finally:
        conn.close()

if __name__ == "__main__":
    main()

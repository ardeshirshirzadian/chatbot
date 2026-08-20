"""One-time migration: reads knowledge/faq.csv (same parsing logic as
load_all_knowledge_bases() in main.py) and inserts each row into the
Postgres `faq` table. Does not touch the CSV file.
"""
import csv
import secrets
import string
from pathlib import Path

import psycopg2

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "user": "chatbot",
    "password": "Xk7#mQ2vN9pL$wR4tZ8j",
    "dbname": "chatbot_faq",
}

FAQ_CSV_PATH = Path(__file__).parent / "knowledge" / "faq.csv"

ID_ALPHABET = string.ascii_uppercase + string.digits


def generate_id(existing_ids):
    while True:
        candidate = "".join(secrets.choice(ID_ALPHABET) for _ in range(8))
        if candidate not in existing_ids:
            existing_ids.add(candidate)
            return candidate


def read_faq_rows(path):
    """Mirrors the FAQ-parsing branch of load_all_knowledge_bases() in main.py."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample) if sample else None
            if dialect and dialect.delimiter in [",", ";", "\t"]:
                reader = csv.DictReader(f, dialect=dialect)
            else:
                reader = csv.DictReader(f)
        except Exception:
            reader = csv.DictReader(f)

        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        if "question" not in headers:
            raise ValueError(f"{path} does not look like an FAQ CSV (no 'question' column)")

        for row in reader:
            clean_row = {k.strip(): v.strip() for k, v in row.items() if k and v}

            question = clean_row.get("Question", clean_row.get("question", "")).strip()
            answer = clean_row.get("Sample_Answer", clean_row.get("answer", "")).strip()
            category = clean_row.get("Category", clean_row.get("category", "عمومی")).strip()

            if question and answer:
                rows.append({"category": category, "question": question, "answer": answer})

    return rows


def main():
    rows = read_faq_rows(FAQ_CSV_PATH)
    print(f"📄 Read {len(rows)} rows from {FAQ_CSV_PATH}")

    existing_ids = set()
    conn = psycopg2.connect(**DB_CONFIG)
    inserted = 0
    skipped = 0
    try:
        with conn:
            with conn.cursor() as cur:
                for row in rows:
                    new_id = generate_id(existing_ids)
                    cur.execute(
                        """
                        INSERT INTO faq (id, category, question, answer, event_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (new_id, row["category"], row["question"], row["answer"], 1),
                    )
                    if cur.rowcount == 1:
                        inserted += 1
                    else:
                        skipped += 1
    finally:
        conn.close()

    print(f"✅ Migration complete: {inserted} rows inserted, {skipped} skipped (conflict).")
    print(f"ℹ️  {FAQ_CSV_PATH} left untouched.")


if __name__ == "__main__":
    main()

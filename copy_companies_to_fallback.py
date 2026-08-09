import psycopg2
import psycopg2.extras
from psycopg2.extras import Json

SRC = {"host": "127.0.0.1", "port": 5432, "user": "iphapp", "password": "LMkAcLyHGYlq7bWr0D4hKSGLImZfwVdo", "dbname": "iphsuperapp"}
DST = {"host": "127.0.0.1", "port": 5434, "user": "chatbot", "password": "Xk7#mQ2vN9pL$wR4tZ8j", "dbname": "chatbot_faq"}

src_conn = psycopg2.connect(**SRC)
dst_conn = psycopg2.connect(**DST)

src_cur = src_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
src_cur.execute("""
    SELECT id, event_id, brand_name_fa, brand_name_en, legal_name_fa, legal_name_en,
           logo, website, description_fa, description_en, slug, phones, emails,
           address_fa, address_en, industry_id, hall_name, booth_no, is_sponsor,
           sponsor_level, booth_uuid, booth_xp, is_manual, linked_mission_id,
           linked_badge_id, repeatable_scan, repeatable_scan_hours, repeatable_start_hour
    FROM companies WHERE is_active = true
""")
rows = src_cur.fetchall()
print(f"Fetched {len(rows)} active companies from main DB")

dst_cur = dst_conn.cursor()
count = 0
for r in rows:
    r = dict(r)
    for jsonb_field in ("logo", "phones", "emails"):
        if r.get(jsonb_field) is not None:
            r[jsonb_field] = Json(r[jsonb_field])

    dst_cur.execute("""
        INSERT INTO companies (
            id, event_id, brand_name_fa, brand_name_en, legal_name_fa, legal_name_en,
            logo, website, description_fa, description_en, slug, phones, emails,
            address_fa, address_en, industry_id, hall_name, booth_no, is_sponsor,
            sponsor_level, booth_uuid, booth_xp, is_manual, linked_mission_id,
            linked_badge_id, repeatable_scan, repeatable_scan_hours, repeatable_start_hour
        ) VALUES (
            %(id)s, %(event_id)s, %(brand_name_fa)s, %(brand_name_en)s, %(legal_name_fa)s, %(legal_name_en)s,
            %(logo)s, %(website)s, %(description_fa)s, %(description_en)s, %(slug)s, %(phones)s, %(emails)s,
            %(address_fa)s, %(address_en)s, %(industry_id)s, %(hall_name)s, %(booth_no)s, %(is_sponsor)s,
            %(sponsor_level)s, %(booth_uuid)s, %(booth_xp)s, %(is_manual)s, %(linked_mission_id)s,
            %(linked_badge_id)s, %(repeatable_scan)s, %(repeatable_scan_hours)s, %(repeatable_start_hour)s
        )
        ON CONFLICT (id) DO UPDATE SET
            event_id=EXCLUDED.event_id, brand_name_fa=EXCLUDED.brand_name_fa, brand_name_en=EXCLUDED.brand_name_en,
            legal_name_fa=EXCLUDED.legal_name_fa, legal_name_en=EXCLUDED.legal_name_en, logo=EXCLUDED.logo,
            website=EXCLUDED.website, description_fa=EXCLUDED.description_fa, description_en=EXCLUDED.description_en,
            slug=EXCLUDED.slug, phones=EXCLUDED.phones, emails=EXCLUDED.emails, address_fa=EXCLUDED.address_fa,
            address_en=EXCLUDED.address_en, industry_id=EXCLUDED.industry_id, hall_name=EXCLUDED.hall_name,
            booth_no=EXCLUDED.booth_no, is_sponsor=EXCLUDED.is_sponsor, sponsor_level=EXCLUDED.sponsor_level,
            booth_uuid=EXCLUDED.booth_uuid, booth_xp=EXCLUDED.booth_xp, is_manual=EXCLUDED.is_manual,
            linked_mission_id=EXCLUDED.linked_mission_id, linked_badge_id=EXCLUDED.linked_badge_id,
            repeatable_scan=EXCLUDED.repeatable_scan, repeatable_scan_hours=EXCLUDED.repeatable_scan_hours,
            repeatable_start_hour=EXCLUDED.repeatable_start_hour
    """, r)
    count += 1

dst_conn.commit()
print(f"Upserted {count} companies into fallback DB")

src_cur.close(); src_conn.close()
dst_cur.close(); dst_conn.close()

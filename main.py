import os
import json
import csv
import re
import secrets
import string
import asyncio
import logging
from typing import Optional
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from contextlib import asynccontextmanager
from functools import lru_cache

import httpx
import numpy as np
import faiss
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from fastapi import FastAPI, Header, HTTPException, Depends, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
#  مسیرها و تنظیمات پایه
# ═══════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
PROMPT_JSON_PATH = BASE_DIR / "prompts" / "system_prompt.json"
EMBEDDINGS_CACHE_PATH = KNOWLEDGE_DIR / "knowledge_embeddings.json"
LOG_FILE_PATH = KNOWLEDGE_DIR / "chat_logs.csv"

OLLAMA_CHAT_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_BASE_URL = OLLAMA_CHAT_URL.replace("/api/chat", "")
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embeddings"

MODEL       = "iranpharma-assistant"
EMBED_MODEL = "bge-m3"

DEFAULT_FALLBACK = "این سؤال خارج از حوزه نمایشگاه ایران‌فارما است یا اطلاعات آن در پایگاه دانش ثبت نشده است."

# ── سقف همزمانی برای بخش‌های وابسته به Ollama (embedding + انتخاب) ──────────
# این سرویس، fallback روی CPU همین VPS است (نه GPU) و همان ۶ هسته را با
# کانتینرهای iph-app/iph-apn/Postgres/Redis شریک است. اندازه‌گیری واقعی روز
# ۲۰۲۶-۰۹-۱۲: دو تولید هم‌زمان روی Ollama (llama-server) به‌تنهایی ~۴.۷ از ۶
# هسته را اشغال کرد. سقف = ۲ هم‌زمان، تا در صورت قطعی سرور GPU طی نمایشگاه،
# ترافیک fallback نتواند بقیه‌ی کانتینرهای همین VPS را گرسنه نگه دارد.
CHAT_CONCURRENCY_LIMIT   = int(os.getenv("CHAT_CONCURRENCY_LIMIT", "2"))
CHAT_QUEUE_WAIT_SECONDS  = float(os.getenv("CHAT_QUEUE_WAIT_SECONDS", "15"))
_chat_llm_semaphore = asyncio.Semaphore(CHAT_CONCURRENCY_LIMIT)

# route.js (iph-app) tries the GPU primary for 15s, then this fallback for up
# to 90s. A 15s queue wait here + a ~30-45s worst-case generation still lands
# well inside that 90s budget, so a busy reply here never collides with
# route.js's own timeout.
CAPACITY_BUSY_MESSAGE = {
    "fa": "چت‌بات موقتاً شلوغ است، لطفاً کمی بعد امتحان کنید.",
    "en": "The chatbot is temporarily busy. Please try again in a moment.",
}

# ── یک httpx.AsyncClient مشترک برای کل اپ ──────────────────────────
# keep-alive + connection pool — در روزهای نمایشگاه فشار کمتری روی Ollama
_http: httpx.AsyncClient = None

# ═══════════════════════════════════════════════
#  حالت‌های Global — یک FAISS index/KB جدا به‌ازای هر local event_id
#  (معماری انتخاب‌شده: هزینه‌ی حافظه‌ی بیشتر به‌جای یک index مشترک با فیلتر
#  metadata — trade-off آگاهانه، فعلاً فقط event_id=1 و 2 داریم)
# ═══════════════════════════════════════════════
KNOWLEDGE_BASE  = {}   # dict[int, list] — {event_id: [item, ...]}
prompt_config   = {}

# ── bot_settings (Postgres) — تنظیمات سراسری bot به‌ازای هر event، admin-editable بدون ری‌استارت ──
# در startup و بعد از هر PUT /admin/bot-settings/{key} دوباره پر می‌شود.
# شکل: {event_id: {key: {"fa": value_fa, "en": value_en}}}
BOT_SETTINGS    = {}
faiss_index     = {}   # dict[int, faiss.IndexFlatIP]
embedding_dimension = {}   # dict[int, int]

# ── English-mode FAISS index (فقط آیتم‌هایی که ترجمه انگلیسی دارند) ──
# faiss_index_en[event_id] روی زیرمجموعه‌ای از KNOWLEDGE_BASE[event_id] ساخته می‌شود؛
# KB_EN_INDICES[event_id][i] اندیس واقعی آیتم در KNOWLEDGE_BASE[event_id] را برای
# نتیجه i-ام جستجوی FAISS انگلیسی همان event برمی‌گرداند.
faiss_index_en       = {}   # dict[int, faiss.IndexFlatIP]
embedding_dimension_en = {}   # dict[int, int]
KB_EN_INDICES        = {}   # dict[int, list]

# قفل نوشتن لاگ — جلوگیری از race condition هنگام درخواست‌های همزمان
_log_lock = asyncio.Lock()


# ═══════════════════════════════════════════════
#  اتصال Postgres — FAQ
# ═══════════════════════════════════════════════
FAQ_DB_HOST     = os.getenv("FAQ_DB_HOST", "127.0.0.1")
FAQ_DB_PORT     = os.getenv("FAQ_DB_PORT", "5433")
FAQ_DB_USER     = os.getenv("FAQ_DB_USER", "chatbot")
FAQ_DB_PASSWORD = os.getenv("FAQ_DB_PASSWORD", "Xk7#mQ2vN9pL$wR4tZ8j")
FAQ_DB_NAME     = os.getenv("FAQ_DB_NAME", "chatbot_faq")

# کلید ادمین برای endpointهای مدیریت FAQ — باید به‌صورت env var ست شود، مقدار پیش‌فرض ندارد
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

FAQ_ID_ALPHABET = string.ascii_uppercase + string.digits


def get_faq_db_connection():
    return psycopg2.connect(
        host=FAQ_DB_HOST,
        port=FAQ_DB_PORT,
        user=FAQ_DB_USER,
        password=FAQ_DB_PASSWORD,
        dbname=FAQ_DB_NAME,
    )


def verify_admin_key(x_admin_key: str | None = Header(None, alias="X-Admin-Key")):
    if not ADMIN_API_KEY or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def wait_for_postgres_ready(max_attempts: int = 15, delay_seconds: int = 2) -> bool:
    """
    قبل از اولین rebuild_knowledge_base در startup تلاش می‌کند به همان Postgres
    که load_bot_settings()/load_all_knowledge_bases() استفاده می‌کنند وصل شود.
    اگر faq-postgres هم‌زمان با اپ ری‌استارت شده باشد (مثلاً بعد از قطع برق
    سرور)، ممکن است چند ثانیه "starting up" بماند و اتصال را رد کند — این تابع
    با یک اتصال سبک و تلاش مجدد منتظر آماده‌شدن آن می‌ماند.
    """
    for attempt in range(1, max_attempts + 1):
        conn = None
        try:
            conn = get_faq_db_connection()
            return True
        except Exception as e:
            print(f"⏳ Waiting for Postgres... attempt {attempt}/{max_attempts} ({e})", flush=True)
            await asyncio.sleep(delay_seconds)
        finally:
            if conn:
                conn.close()
    print("❌ Postgres did not become ready in time.", flush=True)
    return False


# ═══════════════════════════════════════════════
#  توابع کمکی sync (CPU-bound — بدون I/O)
# ═══════════════════════════════════════════════
def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("ۀ", "ه").replace("ة", "ه")
    text = text.replace("‌", " ")
    text = re.sub(r"[^\w\sآ-ی]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_prompt_config():
    try:
        with open(PROMPT_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load prompt config: {e}")
        return {"fallback": DEFAULT_FALLBACK, "examples": []}


def load_bot_settings() -> dict:
    """
    جدول bot_settings را از Postgres می‌خواند — تنظیمات key-value به‌ازای هر
    local event_id (فعلاً فقط fallback_message، ولی برای تنظیمات آینده هم شکل
    مناسبی دارد). PK جدول (event_id, key) است.
    خروجی: {event_id: {key: {"fa": value_fa, "en": value_en}}}
    در startup و بعد از هر PUT /admin/bot-settings/{key} دوباره فراخوانی می‌شود —
    یعنی تغییرات بدون نیاز به ری‌استارت روی درخواست بعدی اعمال می‌شوند.
    """
    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT event_id, key, value_fa, value_en FROM bot_settings")
            rows = cur.fetchall()
    except Exception as e:
        print(f"Warning: Could not load bot_settings: {e}", flush=True)
        return {}
    finally:
        if conn:
            conn.close()
    settings_by_event: dict = {}
    for event_id, key, value_fa, value_en in rows:
        settings_by_event.setdefault(event_id, {})[key] = {"fa": value_fa, "en": value_en}
    return settings_by_event


def get_fallback_message(event_id: int, lang: str) -> str:
    """
    پیام fallback را برای event/زبان درخواست از BOT_SETTINGS برمی‌گرداند.
    اگر مقدار دیتابیس برای این event/زبان خالی/موجود نبود (مثلاً eventـی که
    هنوز هیچ bot_settings ندارد)، DEFAULT_FALLBACK (هاردکد در کد، همیشه در
    دسترس) به‌عنوان آخرین لایه‌ی امن استفاده می‌شود.
    """
    return BOT_SETTINGS.get(event_id, {}).get("fallback_message", {}).get(lang) or DEFAULT_FALLBACK


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def keyword_score(user_message: str, search_text: str) -> float:
    user_words = set(normalize_text(user_message).split())
    faq_words  = set(normalize_text(search_text).split())
    if not user_words or not faq_words:
        return 0.0
    return len(user_words & faq_words) / len(user_words)


def build_search_text(question: str, answer: str, category: str = "", source_file: str = "") -> str:
    return f"منبع: {source_file}\nدسته‌بندی: {category}\nسؤال: {question}\nپاسخ: {answer}".strip()


# ═══════════════════════════════════════════════
#  I/O async — embed و LLM
# ═══════════════════════════════════════════════
async def embed_text_async(text: str) -> list:
    """
    embedding را به‌صورت async از Ollama می‌گیرد.
    در startup به‌صورت موازی (gather) فراخوانی می‌شود.
    در request هر بار یک‌بار await می‌شود — ترتیب حفظ می‌شود.
    """
    resp = await _http.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": normalize_text(text)},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


async def call_ollama_async(messages: list, num_predict: int = 3, temperature: float = 0) -> str | None:
    """
    یک فراخوانی async به Ollama chat API.
    stream=False — فقط یک عدد برمی‌گرداند (انتخاب کاندیدا).
    timeout=30s — اگر مدل کند بود graceful timeout.
    """
    try:
        resp = await _http.post(
            OLLAMA_CHAT_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": num_predict},
            },
            timeout=45.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except httpx.TimeoutException:
        print("⏱️  Ollama timeout", flush=True)
    except Exception as e:
        print(f"⚠️  Ollama error: {e}", flush=True)
    return None


async def warmup_ollama_model():
    """
    یک درخواست حداقلی به مدل chat می‌فرستد تا در startup در حافظه GPU لود شود،
    به‌جای اینکه اولین کاربر واقعی منتظر cold start بماند.
    """
    try:
        return await call_ollama_async(
            messages=[{"role": "user", "content": "hi"}],
            num_predict=1,
            temperature=0,
        )
    except Exception as e:
        print(f"⚠️  Ollama warmup failed: {e}", flush=True)
        return None


# ═══════════════════════════════════════════════
#  لاگ async — بدون race condition
# ═══════════════════════════════════════════════
LOG_HEADER = ["Timestamp", "User_Message", "Bot_Answer", "Source", "Score", "Matched_Question", "Event_Id", "User_Uuid"]


def _migrate_log_header_if_stale():
    """
    اگر chat_logs.csv از قبل با header قدیمی (بدون ستون Event_Id یا User_Uuid)
    وجود دارد، فایل قدیمی را کنار می‌گذارد (rename با timestamp) تا نوشتن بعدی
    یک فایل تازه با header جدید بسازد. فقط یک‌بار لازم است اجرا شود — بعد از
    اولین self-heal، فایل جدید همیشه header درست را دارد. صدا زدن این تابع
    باید زیر _log_lock باشد (توسط caller تضمین می‌شود).
    """
    if not LOG_FILE_PATH.exists():
        return
    try:
        with open(LOG_FILE_PATH, newline="", encoding="utf-8-sig") as f:
            first_line = f.readline()
        if "User_Uuid" not in first_line:
            stamp = datetime.now().strftime("%Y%m%d%H%M%S")
            LOG_FILE_PATH.rename(LOG_FILE_PATH.with_name(f"{LOG_FILE_PATH.stem}.pre-user-uuid-migration-{stamp}.csv"))
    except Exception:
        pass


async def log_chat_interaction(user_msg: str, bot_ans: str, source: str, score: float, matched_q: str = "", event_id: int | None = None, user_uuid: str | None = None):
    async with _log_lock:
        try:
            _migrate_log_header_if_stale()
            file_exists = LOG_FILE_PATH.exists()
            LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(LOG_HEADER)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user_msg, bot_ans, source, score, matched_q, event_id, user_uuid
                ])
        except Exception:
            pass


# ═══════════════════════════════════════════════
#  بارگذاری Knowledge Base (sync — فقط startup / rebuild)
# ═══════════════════════════════════════════════
def load_faq_from_postgres(event_id: int | None = None):
    """
    FAQ items از جدول Postgres `faq` — جایگزین knowledge/faq.csv.
    اگر event_id داده شود فقط ردیف‌های همان event خوانده می‌شوند (rebuild
    تک‌event، برای efficiency)؛ در غیر این صورت همه‌ی ردیف‌های همه‌ی eventها در
    یک کوئری خوانده می‌شوند (rebuild کامل — گروه‌بندی بر اساس event_id در
    load_all_knowledge_bases انجام می‌شود، نه اینجا).
    """
    faq_items = []
    try:
        conn = get_faq_db_connection()
        try:
            with conn.cursor() as cur:
                if event_id is not None:
                    cur.execute(
                        "SELECT id, category, question, answer, question_en, answer_en, event_id "
                        "FROM faq WHERE event_id = %s ORDER BY created_at",
                        (event_id,),
                    )
                else:
                    cur.execute(
                        "SELECT id, category, question, answer, question_en, answer_en, event_id "
                        "FROM faq ORDER BY created_at"
                    )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        print(f"Error loading FAQ from Postgres: {e}", flush=True)
        return faq_items

    for row_id, category, question, answer, question_en, answer_en, row_event_id in rows:
        question = (question or "").strip()
        answer   = (answer or "").strip()
        category = (category or "عمومی").strip()
        question_en = (question_en or "").strip()
        answer_en   = (answer_en or "").strip()
        if question and answer:
            item = {
                "id": row_id,
                "event_id": row_event_id,
                "category": category,
                "question": question,
                "question_norm": normalize_text(question),
                "answer": answer,
                "search_text": build_search_text(question, answer, category, "postgres:faq"),
                "source_file": "postgres:faq",
                "is_directory": False,
                # ── فیلدهای انگلیسی (اختیاری) — فقط وقتی هر دو question_en/answer_en
                #    پر باشند این آیتم در جستجوی lang=en قابل تطبیق می‌شود ──
                "question_en": question_en,
                "answer_en": answer_en,
                "question_en_norm": normalize_text(question_en) if (question_en and answer_en) else "",
                "search_text_en": (
                    build_search_text(question_en, answer_en, category, "postgres:faq")
                    if (question_en and answer_en) else ""
                ),
            }
            faq_items.append(item)

    return faq_items


def _format_jsonish_field(value) -> str:
    """نمایش خوانا از یک ستون jsonb (لیست/دیکشنری) — برای phones/emails/logo."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(", ".join(str(v) for v in item.values() if v))
            elif item:
                parts.append(str(item))
        return "، ".join(p for p in parts if p)
    if isinstance(value, dict):
        return ", ".join(str(v) for v in value.values() if v)
    return str(value)


def load_companies_from_postgres(event_id: int | None = None):
    """
    آیتم‌های دایرکتوری شرکت‌ها از جدول Postgres `companies` — جایگزین
    knowledge/companies.csv. اگر event_id داده شود فقط شرکت‌های همان event
    خوانده می‌شوند (rebuild تک‌event)؛ در غیر این صورت همه‌ی eventها در یک
    کوئری خوانده می‌شوند (گروه‌بندی در load_all_knowledge_bases).
    """
    company_items = []
    try:
        conn = get_faq_db_connection()
        try:
            with conn.cursor() as cur:
                base_query = (
                    "SELECT id, brand_name_fa, brand_name_en, hall_name, booth_no, website, "
                    "phones, emails, address_fa, address_en, description_en, event_id "
                    "FROM companies"
                )
                if event_id is not None:
                    cur.execute(base_query + " WHERE event_id = %s", (event_id,))
                else:
                    cur.execute(base_query)
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        print(f"Error loading companies from Postgres: {e}", flush=True)
        return company_items

    for (row_id, brand_name_fa, brand_name_en, hall_name, booth_no, website,
         phones, emails, address_fa, address_en, description_en, row_event_id) in rows:
        company_name = (brand_name_fa or "").strip()
        if not company_name:
            continue
        booth_no = (booth_no or "").strip()

        question = (
            f"اطلاعات غرفه و تماس شرکت {company_name} چیست؟ "
            f"{company_name} هست؟ غرفه {company_name} کجاست؟ "
            f"شماره غرفه {company_name} "
            f"آیا {company_name} در نمایشگاه حضور دارد؟"
        )

        extra_parts = []
        if hall_name:
            extra_parts.append(f"• سالن: {hall_name}")
        if booth_no:
            extra_parts.append(f"• شماره غرفه: {booth_no}")
        if website:
            extra_parts.append(f"• وبسایت: {website}")
        phones_str = _format_jsonish_field(phones)
        if phones_str:
            extra_parts.append(f"• تلفن: {phones_str}")
        emails_str = _format_jsonish_field(emails)
        if emails_str:
            extra_parts.append(f"• ایمیل: {emails_str}")
        if address_fa:
            extra_parts.append(f"• آدرس: {address_fa}")

        answer = json.dumps(
            {"company": company_name, "booth": booth_no, "extra": "\n".join(extra_parts)},
            ensure_ascii=False
        )
        category = "دایرکتوری شرکت‌ها و غرفه‌ها"

        item = {
            "id": f"company_{row_id}",
            "event_id": row_event_id,
            "category": category,
            "question": question,
            "question_norm": normalize_text(question),
            "answer": answer,
            "search_text": build_search_text(question, answer, category, "postgres:companies"),
            "source_file": "postgres:companies",
            "is_directory": True,
        }

        # ── نمایندگی انگلیسی (اختیاری) — فقط اگر brand_name_en موجود باشد.
        #    فیلدهای فارسیِ متنی (مثل description_fa) هرگز اینجا استفاده نمی‌شوند؛
        #    فیلدهای خنثی از نظر زبان (website/phones/hall_name/booth_no) در هر دو حالت قابل استفاده‌اند. ──
        company_name_en = (brand_name_en or "").strip()
        if company_name_en:
            question_en = (
                f"What are the booth and contact details of {company_name_en}? "
                f"Is {company_name_en} present at the exhibition? "
                f"Where is the {company_name_en} booth? "
                f"Booth number for {company_name_en}"
            )

            extra_parts_en = []
            if hall_name:
                extra_parts_en.append(f"• Hall: {hall_name}")
            if booth_no:
                extra_parts_en.append(f"• Booth No: {booth_no}")
            if website:
                extra_parts_en.append(f"• Website: {website}")
            if phones_str:
                extra_parts_en.append(f"• Phone: {phones_str}")
            if emails_str:
                extra_parts_en.append(f"• Email: {emails_str}")
            address_en_clean = (address_en or "").strip()
            if address_en_clean:
                extra_parts_en.append(f"• Address: {address_en_clean}")
            description_en_clean = (description_en or "").strip()
            if description_en_clean:
                extra_parts_en.append(f"• About: {description_en_clean}")

            answer_en = json.dumps(
                {"company": company_name_en, "booth": booth_no, "extra": "\n".join(extra_parts_en)},
                ensure_ascii=False
            )

            item["question_en"] = question_en
            item["question_en_norm"] = normalize_text(question_en)
            item["answer_en"] = answer_en
            item["search_text_en"] = build_search_text(question_en, answer_en, category, "postgres:companies")
        else:
            item["question_en"] = ""
            item["question_en_norm"] = ""
            item["answer_en"] = ""
            item["search_text_en"] = ""

        company_items.append(item)

    return company_items


def _format_panel_time(starts_at, ends_at) -> str:
    """starts_at/ends_at از Postgres به‌صورت naive datetime می‌آیند — همان
    مقداری که بقیه‌ی اپ (مثل صفحه‌ی عمومی پنل‌ها) بدون تبدیل timezone نمایش
    می‌دهد؛ اینجا هم بدون تبدیل فرمت می‌شوند، فقط برای خوانایی."""
    if not starts_at:
        return ""
    try:
        date_str = starts_at.strftime("%Y-%m-%d")
        start_str = starts_at.strftime("%H:%M")
        if ends_at:
            return f"{date_str} ساعت {start_str} تا {ends_at.strftime('%H:%M')}"
        return f"{date_str} ساعت {start_str}"
    except Exception:
        return ""


def _short_title(t: str, max_len: int = 45) -> str:
    """
    Selector-display-only title shortener. Panel/workshop titles that
    combine a short headline with a longer elaboration are consistently
    authored with a Persian semicolon between the two parts (e.g. "X؛
    توضیح بیشتر درباره X"). Prefer that natural split; a title with
    neither a semicolon nor natural brevity falls back to a hard
    word-boundary character cap. Only feeds question_display/_en -- the
    real title is untouched everywhere else (question, search_text,
    answer, and anything shown to the user).
    """
    if "؛" in t:
        t = t.split("؛", 1)[0].strip()
    if len(t) > max_len:
        truncated = t[:max_len].rsplit(" ", 1)[0]
        t = (truncated or t[:max_len]) + "…"
    return t


def load_panels_from_postgres(event_id: int | None = None):
    """
    آیتم‌های پنل/کارگاه از جدول Postgres `panels` — جایگزین knowledge_list
    قدیمی مشابه companies، اما بدون is_directory: پاسخ‌ها همان‌جا در ingestion
    به‌صورت متن آماده ساخته می‌شوند (نه JSON نیازمند فرمت‌دهی جدا مثل
    format_directory_response شرکت‌ها) چون شکل سوال‌های پنل/کارگاه یکنواخت‌تر
    است. kind (PANEL/WORKSHOP، هرچند این ستون در دیتای منبع بین حروف بزرگ و
    کوچک ناسازگار است) فقط برای انتخاب دسته‌بندی/برچسب استفاده می‌شود — یک
    نوع آیتم KB جدا نیست.
    """
    panel_items = []
    try:
        conn = get_faq_db_connection()
        try:
            with conn.cursor() as cur:
                base_query = (
                    "SELECT id, title_fa, title_en, description_fa, description_en, "
                    "hall_fa, hall_en, starts_at, ends_at, kind, speakers, event_id "
                    "FROM panels"
                )
                if event_id is not None:
                    cur.execute(base_query + " WHERE event_id = %s", (event_id,))
                else:
                    cur.execute(base_query)
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        print(f"Error loading panels from Postgres: {e}", flush=True)
        return panel_items

    for (row_id, title_fa, title_en, description_fa, description_en,
         hall_fa, hall_en, starts_at, ends_at, kind, speakers, row_event_id) in rows:
        title = (title_fa or "").strip()
        if not title:
            continue

        is_workshop = (kind or "").strip().upper() == "WORKSHOP"
        category = "کارگاه‌های آموزشی نمایشگاه" if is_workshop else "پنل‌های نمایشگاه"
        kind_label = "کارگاه" if is_workshop else "پنل"
        time_str = _format_panel_time(starts_at, ends_at)

        question = (
            f"{kind_label} {title} چه زمانی برگزار می‌شود؟ "
            f"{title} کجاست؟ در چه سالنی برگزار می‌شود؟ "
            f"سخنرانان {title} چه کسانی هستند؟ "
            f"درباره {kind_label} {title} توضیح بده"
        )
        # Selector-facing display text: one clean sentence, not the four-part
        # run-on question above. select_best_candidate()'s LLM judge is asked
        # whether an option "DIRECTLY and SPECIFICALLY" answers the user's
        # question; a multi-question blob reads as ambiguous to it and gets
        # rejected even when it's the correct top FAISS match. `question` and
        # `search_text` (the rich blob) still drive embedding/exact-match,
        # unchanged.
        question_display = f"{kind_label} {_short_title(title)} چه زمانی و در کجا برگزار می‌شود؟"

        speakers_fa = []
        for sp in (speakers or []):
            name = f"{(sp.get('firstname_fa') or '').strip()} {(sp.get('lastname_fa') or '').strip()}".strip()
            if not name:
                continue
            job = (sp.get('job_title_fa') or '').strip()
            speakers_fa.append(f"{name} ({job})" if job else name)

        answer_parts = [f"🎤 {kind_label}: {title}"]
        if time_str:
            answer_parts.append(f"🕒 زمان: {time_str}")
        if hall_fa:
            answer_parts.append(f"📍 سالن: {hall_fa}")
        if speakers_fa:
            answer_parts.append(f"🎙️ سخنران(ها): {', '.join(speakers_fa)}")
        description_fa_s = (description_fa or "").strip()
        if description_fa_s:
            answer_parts.append(f"\n{description_fa_s}")
        answer = "\n".join(answer_parts)

        # ── نسخه انگلیسی (اختیاری) — فقط اگر title_en موجود باشد، همان الگوی companies
        title_en_s = (title_en or "").strip()
        question_en = answer_en = question_en_norm = search_text_en = question_display_en = None
        if title_en_s:
            kind_label_en = "Workshop" if is_workshop else "Panel"
            question_en = (
                f"When is the {kind_label_en.lower()} {title_en_s}? "
                f"Where is {title_en_s}? Which hall is it in? "
                f"Who are the speakers at {title_en_s}? "
                f"Tell me about {title_en_s}"
            )
            question_display_en = f"When and where is the {kind_label_en.lower()} {_short_title(title_en_s)} held?"

            speakers_en = []
            for sp in (speakers or []):
                name_en = f"{(sp.get('firstname_en') or '').strip()} {(sp.get('lastname_en') or '').strip()}".strip()
                if name_en:
                    speakers_en.append(name_en)

            answer_parts_en = [f"🎤 {kind_label_en}: {title_en_s}"]
            if time_str:
                answer_parts_en.append(f"🕒 Time: {time_str}")
            if hall_en:
                answer_parts_en.append(f"📍 Hall: {hall_en}")
            if speakers_en:
                answer_parts_en.append(f"🎙️ Speaker(s): {', '.join(speakers_en)}")
            description_en_s = (description_en or "").strip()
            if description_en_s:
                answer_parts_en.append(f"\n{description_en_s}")
            answer_en = "\n".join(answer_parts_en)

            question_en_norm = normalize_text(question_en)
            search_text_en = build_search_text(question_en, answer_en, category, "postgres:panels")

        panel_items.append({
            "id": f"panel_{row_id}",
            "event_id": row_event_id,
            "category": category,
            "question": question,
            "question_display": question_display,
            "question_norm": normalize_text(question),
            "answer": answer,
            "search_text": build_search_text(question, answer, category, "postgres:panels"),
            "source_file": "postgres:panels",
            "is_directory": False,
            "question_en": question_en,
            "question_display_en": question_display_en,
            "answer_en": answer_en,
            "question_en_norm": question_en_norm,
            "search_text_en": search_text_en,
        })

    return panel_items


def load_all_knowledge_bases(event_id: int | None = None) -> dict:
    """
    خروجی: {event_id: [item, ...]} — یک KB جدا به‌ازای هر local event_id.
    اگر event_id داده شود، فقط همان event از Postgres خوانده می‌شود (rebuild
    تک‌event، سریع‌تر)؛ در غیر این صورت همه‌ی ردیف‌های همه‌ی eventها در یک
    کوئری خوانده و اینجا در پایتون بر اساس event_id گروه‌بندی می‌شوند
    (rebuild کامل — به‌جای N کوئری جدا به‌ازای هر event).
    فایل‌های CSV قدیمی (اگر باقی مانده باشند) هیچ event_id‌ای ندارند — به‌طور
    پیش‌فرض به event_id=1 نسبت داده می‌شوند.
    """
    knowledge_list = load_faq_from_postgres(event_id)
    knowledge_list.extend(load_companies_from_postgres(event_id))
    knowledge_list.extend(load_panels_from_postgres(event_id))

    csv_items = []
    if KNOWLEDGE_DIR.exists() and (event_id is None or event_id == 1):
        csv_items = _load_csv_knowledge_items()

    by_event: dict = {}
    for item in knowledge_list:
        by_event.setdefault(item["event_id"], []).append(item)
    for item in csv_items:
        by_event.setdefault(1, []).append(item)

    return by_event


def _load_csv_knowledge_items() -> list:
    """CSV fallback (companies.csv/faq.csv دیگر استفاده نمی‌شوند، فقط فایل‌های
    دیگر) — بدون مفهوم event، همیشه به event_id=1 نسبت داده می‌شود (در
    load_all_knowledge_bases)."""
    knowledge_list = []
    # faq.csv و companies.csv دیگر خوانده نمی‌شوند — هر دو اکنون از Postgres می‌آیند.
    # هر فایل CSV دیگری (در صورت وجود) طبق منطق قبلی پردازش می‌شود.
    csv_files = [
        f for f in KNOWLEDGE_DIR.glob("*.csv")
        if f.name not in ("chat_logs.csv", "faq.csv", "companies.csv")
    ]

    for file_path in csv_files:
        try:
            with open(file_path, newline="", encoding="utf-8-sig") as f:
                sample = f.read(2048); f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample) if sample else None
                    if dialect and dialect.delimiter in [',', ';', '\t']:
                        reader = csv.DictReader(f, dialect=dialect)
                    else:
                        reader = csv.DictReader(f)
                except Exception:
                    reader = csv.DictReader(f)
                headers = [h.strip().lower() for h in (reader.fieldnames or [])]
                is_faq  = "question" in headers

                file_rows_count = 0
                for row in reader:
                    clean_row = {k.strip(): v.strip() for k, v in row.items() if k and v}

                    if is_faq:
                        question = clean_row.get("Question", clean_row.get("question", "")).strip()
                        answer   = clean_row.get("Sample_Answer", clean_row.get("answer", "")).strip()
                        category = clean_row.get("Category", clean_row.get("category", "عمومی")).strip()
                        is_dir   = False
                    else:
                        company_name = (
                            clean_row.get("نام شرکت") or clean_row.get("نام")
                            or clean_row.get("company") or clean_row.get("Company_Name")
                            or (list(clean_row.values())[0] if clean_row else "")
                        )
                        if not company_name:
                            continue
                        booth_no = (
                            clean_row.get("شماره غرفه") or clean_row.get("غرفه")
                            or clean_row.get("booth") or clean_row.get("Booth_No", "")
                        )
                        question = (
                            f"اطلاعات غرفه و تماس شرکت {company_name} چیست؟ "
                            f"{company_name} هست؟ غرفه {company_name} کجاست؟ "
                            f"شماره غرفه {company_name} "
                            f"آیا {company_name} در نمایشگاه حضور دارد؟"
                        )
                        extra_keys  = {"نام شرکت","نام","company","Company_Name","شماره غرفه","غرفه","booth","Booth_No"}
                        extra_parts = [f"• {k}: {v}" for k, v in clean_row.items() if k not in extra_keys]
                        answer   = json.dumps(
                            {"company": company_name, "booth": booth_no, "extra": "\n".join(extra_parts)},
                            ensure_ascii=False
                        )
                        category = "دایرکتوری شرکت‌ها و غرفه‌ها"
                        is_dir   = True

                    if question and answer:
                        knowledge_list.append({
                            "id": clean_row.get("ID", clean_row.get("id", str(file_rows_count))),
                            "category": category,
                            "question": question,
                            "question_norm": normalize_text(question),
                            "answer": answer,
                            "search_text": build_search_text(question, answer, category, file_path.name),
                            "source_file": file_path.name,
                            "is_directory": is_dir,
                        })
                        file_rows_count += 1
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}", flush=True)

    return knowledge_list


# ═══════════════════════════════════════════════
#  Rebuild Knowledge Base + FAISS — startup و بعد از هر تغییر admin
#  یک KB/FAISS جدا به‌ازای هر local event_id — نه یک index مشترک با فیلتر.
# ═══════════════════════════════════════════════
def _build_faiss_for_items(kb_items: list, cached_embeddings: dict):
    """
    FAISS index (+ نسخه‌ی انگلیسی) را برای لیست آیتم‌های یک event می‌سازد،
    با استفاده از cached_embeddings مشترک (کلید = متن نرمال‌شده، بین eventها
    به اشتراک گذاشته می‌شود — دو event با متن سؤال یکسان می‌توانند از یک
    embedding استفاده کنند، کاملاً بی‌خطر).
    خروجی: (faiss_index, embedding_dimension, faiss_index_en, embedding_dimension_en, kb_en_indices)
    """
    embedding_list = [
        cached_embeddings[item["question_norm"]]
        for item in kb_items
        if item["question_norm"] in cached_embeddings
    ]
    if embedding_list:
        emb_np = np.array(embedding_list).astype("float32")
        dim = emb_np.shape[1]
        faiss.normalize_L2(emb_np)
        index = faiss.IndexFlatIP(dim)
        index.add(emb_np)
    else:
        index, dim = None, None

    embedding_list_en = []
    kb_en_indices = []
    for i, item in enumerate(kb_items):
        en_norm = item.get("question_en_norm")
        if en_norm and en_norm in cached_embeddings:
            embedding_list_en.append(cached_embeddings[en_norm])
            kb_en_indices.append(i)

    if embedding_list_en:
        emb_np_en = np.array(embedding_list_en).astype("float32")
        dim_en = emb_np_en.shape[1]
        faiss.normalize_L2(emb_np_en)
        index_en = faiss.IndexFlatIP(dim_en)
        index_en.add(emb_np_en)
    else:
        index_en, dim_en = None, None

    return index, dim, index_en, dim_en, kb_en_indices


async def rebuild_knowledge_base(event_id: int | None = None) -> bool:
    """
    KNOWLEDGE_BASE[event] را از Postgres (FAQ+companies) + CSVها دوباره می‌سازد،
    فقط آیتم‌های جدید/تغییریافته را embed می‌کند (با استفاده از cache مشترک) و
    FAISS index هر event را از نو می‌سازد. global های دیکشنری‌شده‌ای که /chat
    استفاده می‌کند به‌روز می‌شوند.

    event_id=None (پیش‌فرض؛ lifespan startup و self-heal استفاده می‌کنند):
        rebuild کامل — همه‌ی eventهای موجود در داده از نو ساخته می‌شوند؛
        eventـی که دیگر هیچ ردیفی ندارد به لیست خالی می‌رسد (نه اینکه داده‌ی
        قدیمی‌اش برای همیشه بماند).
    event_id=<int> (endpointهای admin بعد از یک نوشتن موفق در Postgres):
        فقط KB/FAISS همان یک event دوباره ساخته می‌شود — سریع‌تر، به بقیه‌ی
        eventها دست نمی‌زند.

    همه چیز ابتدا در متغیرهای local ساخته می‌شود؛ global ها فقط یک‌جا و فقط در
    انتها — بعد از ساخته‌شدن کامل و بدون خطای هر چیزی — جایگزین می‌شوند. اگر در
    هر نقطه‌ای (ردیف خراب Postgres، خطای پیش‌بینی‌نشده‌ی embedding، خطای numpy/faiss)
    استثنایی رخ دهد، global های قبلی دست‌نخورده باقی می‌مانند.

    Returns:
        True  اگر rebuild کامل و موفق بود (global ها به‌روزرسانی شدند).
        False اگر rebuild شکست خورد (global های قبلی دست‌نخورده ماندند).
    """
    global KNOWLEDGE_BASE, faiss_index, embedding_dimension
    global faiss_index_en, embedding_dimension_en, KB_EN_INDICES

    try:
        new_kb_by_event = load_all_knowledge_bases(event_id)
        total_items = sum(len(v) for v in new_kb_by_event.values())
        scope = f"event_id={event_id}" if event_id is not None else "full rebuild"
        print(f"✅ Knowledge base loaded: {total_items} items across {len(new_kb_by_event)} event(s) ({scope})", flush=True)

        if event_id is None:
            # rebuild کامل — اگر نتیجه‌ی جدید کاملاً خالی است ولی KNOWLEDGE_BASE
            # فعلی خالی نبود، این احتمالاً یک قطعی موقت Postgres است (که
            # load_faq_from_postgres/load_companies_from_postgres بی‌صدا catch و
            # به [] تبدیل می‌کنند، بدون raise) — رفتار قبلی حفظ می‌شود: کل
            # rebuild را شکست‌خورده حساب کن تا داده‌ی سالم با نتیجه‌ی خالی
            # جایگزین نشود. این چک عمداً روی مجموع کل eventهاست، نه تک‌تک
            # eventها — یک event که واقعاً همه‌ی FAQهایش حذف شده نباید کل
            # rebuild چندeventی را متوقف کند.
            current_total = sum(len(v) for v in KNOWLEDGE_BASE.values())
            if total_items == 0 and current_total > 0:
                raise RuntimeError(
                    f"load_all_knowledge_bases() returned 0 items total while current "
                    f"KNOWLEDGE_BASE has {current_total} across {len(KNOWLEDGE_BASE)} event(s) — "
                    f"refusing to replace working data with an empty result"
                )
            # rebuild کامل مرجع همه‌ی eventهای شناخته‌شده است.
            for ev in KNOWLEDGE_BASE:
                new_kb_by_event.setdefault(ev, [])
        else:
            # rebuild تک‌event همیشه بلافاصله بعد از یک نوشتن موفق در همان event
            # در Postgres صدا زده می‌شود (از endpointهای admin) — یعنی Postgres
            # همین الان در دسترس بوده، پس نتیجه‌ی خالی اینجا یک قطعی مشکوک
            # نیست، یک وضعیت واقعی است (مثلاً آخرین FAQ آن event حذف شده) —
            # بدون گارد اعمال می‌شود.
            new_kb_by_event.setdefault(event_id, [])

        # ── بارگذاری cache (مشترک بین همه‌ی eventها) ──
        cached_embeddings = {}
        if EMBEDDINGS_CACHE_PATH.exists():
            try:
                with open(EMBEDDINGS_CACHE_PATH, "r", encoding="utf-8") as f:
                    cached_embeddings = json.load(f)
            except Exception:
                pass

        # ── embedding موازی برای آیتم‌های جدید (روی همه‌ی eventهای این rebuild، یکجا) ──
        all_new_items = [item for items in new_kb_by_event.values() for item in items]
        keys_to_embed = []
        for item in all_new_items:
            if item["question_norm"] not in cached_embeddings:
                keys_to_embed.append(item["question_norm"])
            en_norm = item.get("question_en_norm")
            if en_norm and en_norm not in cached_embeddings:
                keys_to_embed.append(en_norm)
        keys_to_embed = list(dict.fromkeys(keys_to_embed))  # حذف تکراری، ترتیب حفظ می‌شود

        if keys_to_embed:
            print(f"🔄 Embedding {len(keys_to_embed)} new items (parallel)...", flush=True)
            # batch ها را گروه‌بندی کن — ۱۰ تایی تا Ollama اشباع نشود
            BATCH = 10
            for batch_start in range(0, len(keys_to_embed), BATCH):
                batch = keys_to_embed[batch_start: batch_start + BATCH]
                results = await asyncio.gather(
                    *[embed_text_async(key) for key in batch],
                    return_exceptions=True
                )
                for key, result in zip(batch, results):
                    if isinstance(result, Exception):
                        print(f"⚠️  Embedding error for '{key[:50]}': {result}", flush=True)
                    else:
                        cached_embeddings[key] = result

        # ذخیره cache
        try:
            EMBEDDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(EMBEDDINGS_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cached_embeddings, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # ── ساخت FAISS index به‌ازای هر event ──
        new_indices, new_dims = {}, {}
        new_indices_en, new_dims_en, new_kb_en_by_event = {}, {}, {}
        for ev, items in new_kb_by_event.items():
            idx, dim, idx_en, dim_en, en_indices = _build_faiss_for_items(items, cached_embeddings)
            new_indices[ev] = idx
            new_dims[ev] = dim
            new_indices_en[ev] = idx_en
            new_dims_en[ev] = dim_en
            new_kb_en_by_event[ev] = en_indices
            print(
                f"✅ event_id={ev}: FAISS {idx.ntotal if idx else 0} vectors, "
                f"EN {idx_en.ntotal if idx_en else 0} vectors ({len(items)} items)",
                flush=True,
            )

        # ── همه چیز بدون خطا ساخته شد — حالا و فقط حالا global ها را جایگزین کن ──
        # فقط کلیدهای موجود در new_kb_by_event عوض می‌شوند — در rebuild تک‌event
        # بقیه‌ی eventها کاملاً دست‌نخورده می‌مانند.
        KNOWLEDGE_BASE.update(new_kb_by_event)
        faiss_index.update(new_indices)
        embedding_dimension.update(new_dims)
        faiss_index_en.update(new_indices_en)
        embedding_dimension_en.update(new_dims_en)
        KB_EN_INDICES.update(new_kb_en_by_event)
        return True

    except Exception as e:
        logger.error(f"rebuild_knowledge_base failed, keeping previous state: {e}", exc_info=True)
        return False


async def _self_heal_knowledge_base(max_attempts: int = 20, interval_seconds: int = 30):
    """
    Safety net: اگر بعد از rebuild_knowledge_base در startup (حتی بعد از
    wait_for_postgres_ready) هیچ eventـی KB غیرخالی نداشته باشد — به هر دلیلی،
    نه فقط کندی Postgres — این تابع در background هر ۳۰ ثانیه یک‌بار دوباره
    rebuild_knowledge_base (کامل) را امتحان می‌کند تا چت‌بات بدون نیاز به دخالت
    دستی ادمین خودش را ظرف چند دقیقه ترمیم کند.
    """
    for attempt in range(1, max_attempts + 1):
        await asyncio.sleep(interval_seconds)
        print(f"🔄 Self-heal retry {attempt}/{max_attempts}: retrying rebuild_knowledge_base...", flush=True)
        await rebuild_knowledge_base()
        if any(len(kb) > 0 for kb in KNOWLEDGE_BASE.values()):
            total = sum(len(kb) for kb in KNOWLEDGE_BASE.values())
            print(f"✅ Self-heal succeeded on attempt {attempt}: knowledge base has {total} items across {len(KNOWLEDGE_BASE)} event(s).", flush=True)
            return
    print(f"❌ Self-heal gave up after {max_attempts} attempts — knowledge base still empty. Manual admin action may be required.", flush=True)


# ═══════════════════════════════════════════════
#  Lifespan — startup / shutdown
# ═══════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global BOT_SETTINGS, prompt_config, _http

    # ── ساخت httpx client با connection pool ──
    _http = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        timeout=httpx.Timeout(60.0),
    )

    prompt_config = load_prompt_config()
    BOT_SETTINGS  = load_bot_settings()

    # ── منتظر آماده‌شدن Postgres (مثلاً بعد از ری‌استارت هم‌زمان با اپ) قبل از rebuild اول ──
    postgres_ready = await wait_for_postgres_ready()
    if postgres_ready:
        print("✅ Postgres is ready.", flush=True)
    else:
        print("⚠️  Postgres not ready after max wait — proceeding with rebuild anyway.", flush=True)

    await rebuild_knowledge_base()

    # ── safety net: اگر هیچ eventـی KB غیرخالی ندارد، هر ۳۰ ثانیه در background دوباره امتحان کن ──
    if not any(len(kb) > 0 for kb in KNOWLEDGE_BASE.values()):
        print("⚠️  Knowledge base is empty after startup rebuild — scheduling background self-heal retries.", flush=True)
        asyncio.create_task(_self_heal_knowledge_base())

    # ── گرم کردن مدل chat در Ollama (لود در GPU قبل از اولین درخواست) ──
    print("🔥 Warming up Ollama chat model...", flush=True)
    warmup_result = await warmup_ollama_model()
    if warmup_result is not None:
        print("✅ Ollama chat model warmed up", flush=True)
    else:
        print("⚠️  Ollama chat model warmup failed (Ollama may be slow or unavailable)", flush=True)

    yield

    # ── shutdown ──
    await _http.aclose()
    print("✅ HTTP client closed.", flush=True)


# ═══════════════════════════════════════════════
#  FastAPI App
# ═══════════════════════════════════════════════
app = FastAPI(title="Iran Pharma Chat API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    by_event = {str(ev): len(kb) for ev, kb in KNOWLEDGE_BASE.items()}
    return {
        "status": "ok",
        "knowledge_base_items": sum(by_event.values()),
        "knowledge_base_items_by_event": by_event,
    }


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    lang: str = "fa"
    event_id: int
    # Populated server-side by iph-app's /api/chat proxy from the caller's
    # iph_user cookie (see grantChatMissionXp.js's getUserUuid() for the same
    # pattern) -- absent for guest/unauthenticated chat, which is expected,
    # not an error. Never trust this as an auth signal, it's unauthenticated
    # client input forwarded as-is -- log/display only.
    user_uuid: str | None = None


class FAQCreate(BaseModel):
    id: Optional[str] = None
    category: str | None = None
    question: str
    answer: str
    question_en: str | None = None
    answer_en: str | None = None
    synced_to_primary: bool = True


class FAQUpdate(BaseModel):
    category: str | None = None
    question: str | None = None
    answer: str | None = None
    question_en: str | None = None
    answer_en: str | None = None
    synced_to_primary: bool | None = None


class BotSettingUpdate(BaseModel):
    value_fa: str | None = None
    value_en: str | None = None


# ═══════════════════════════════════════════════
#  Intent Detection — rule-based
#  تبدیل سوال عامیانه به کوئری معنادار
# ═══════════════════════════════════════════════
INTENT_PATTERNS = [
    # مسیر و دسترسی
    {
        "keywords": ["بیام", "برم", "بریم", "برسم", "بیاییم", "چطور بیام", "چجوری بیام",
                     "مسیر", "راه", "دسترسی", "چطوری بیام", "چجور بیام",
                     "اومدن", "رفتن", "چطوری میشه اومد", "چجوری میشه اومد"],
        "intent": "مسیر دسترسی به نمایشگاه ایران فارما مصلای بزرگ تهران",
    },
    # آدرس و محل
    {
        "keywords": ["کجاست", "کجاس", "کجا هست", "آدرس", "محل", "مکان", "کجا برگزار",
                     "کجا هستن", "واقع", "موقعیت", "کجا داره", "کجا هه"],
        "intent": "آدرس و محل برگزاری نمایشگاه ایران فارما",
    },
    # تاریخ و زمان
    {
        "keywords": ["کِی", "چه وقت", "چه زمانی", "تاریخ", "ساعت", "روز", "ماه",
                     "برگزار میشه", "برگزار می‌شود", "شروع", "پایان", "اتمام",
                     "تا کی", "از کی", "چند روز", "چه ماهی", "کِی شروع", "کِی تموم"],
        "intent": "تاریخ و زمان برگزاری نمایشگاه ایران فارما",
    },
    # هزینه و بلیط
    {
        "keywords": ["چقدر", "هزینه", "قیمت", "رایگان", "مجانی", "بلیط", "کارت",
                     "ورودیه", "پول", "تعرفه", "هزینه ورود", "پولیه", "ورودی داره"],
        "intent": "هزینه ورود و دریافت کارت ورودی نمایشگاه ایران فارما",
    },
    # ثبت‌نام
    {
        "keywords": ["ثبت‌نام", "ثبت نام", "عضویت", "register", "چطور ثبت",
                     "نحوه ثبت", "فرم", "لینک ثبت", "چطوری ثبت نام کنم",
                     "چجوری ثبت نام", "ثبت نامم", "ثبت‌نامم"],
        "intent": "نحوه ثبت‌نام در نمایشگاه ایران فارما",
    },
    # کارت ورودی
    {
        "keywords": ["کارت ورود", "کارت بازدید", "کارت نمایشگاه", "کارتم",
                     "کارت دریافت", "کارت رو", "کارتو", "کارت مجازی"],
        "intent": "نحوه دریافت کارت ورودی نمایشگاه ایران فارما",
    },
    # مخاطبان و بازدیدکنندگان
    {
        "keywords": ["کی", "کیا", "کیها", "چه کسانی", "چه کسی", "کدوم آدم",
                     "مخاطب", "بازدیدکننده", "بازدید کننده", "کی میاد",
                     "کیا میان", "چه افرادی", "چه کسایی", "کی هستن", "کیا هستن",
                     "کی حضور", "کیا حضور", "چه آدمایی"],
        "intent": "مخاطبان و بازدیدکنندگان نمایشگاه ایران فارما",
    },
    # شرکت‌های حاضر
    {
        "keywords": ["شرکت حضور", "حضور دارن", "حضور دارند", "شرکت‌ها هستن",
                     "کدوم شرکت", "چه شرکتایی", "چه شرکت‌هایی", "شرکت هست",
                     "شرکت میاد", "غرفه داره", "غرفه دارن"],
        "intent": "شرکت‌های حاضر در نمایشگاه ایران فارما",
    },
    # غرفه و اطلاعات شرکت
    {
        "keywords": ["غرفه", "غرفه‌ها", "سالن", "بخش", "شماره غرفه",
                     "غرفه‌دار", "غرفه کجاست", "سالن کجاست"],
        "intent": "اطلاعات غرفه‌ها و سالن‌های نمایشگاه ایران فارما",
    },
    # رزرو غرفه
    {
        "keywords": ["غرفه بگیرم", "غرفه بگیریم", "اجاره غرفه", "رزرو غرفه",
                     "غرفه‌دار بشم", "چطور شرکت کنم", "نمایشگاه‌گذار",
                     "حضور داشته باشم", "غرفه میخوام"],
        "intent": "نحوه اخذ و رزرو غرفه در نمایشگاه ایران فارما",
    },
    # پارکینگ و حمل‌ونقل
    {
        "keywords": ["پارکینگ", "ماشین", "خودرو", "مترو", "اتوبوس", "تاکسی",
                     "حمل‌ونقل", "ایستگاه", "متروی", "اتوبوسی", "پارک کنم",
                     "ماشین بزارم", "ماشینم بزارم"],
        "intent": "پارکینگ و حمل‌ونقل عمومی نمایشگاه ایران فارما",
    },
    # هتل و اقامت
    {
        "keywords": ["هتل", "اقامت", "مهمانپذیر", "اقامتگاه", "هتل نزدیک",
                     "جای موندن", "کجا بمونم", "هتل پیشنهاد"],
        "intent": "هتل‌های نزدیک به محل برگزاری نمایشگاه ایران فارما",
    },
    # تماس و برگزارکننده
    {
        "keywords": ["تماس", "ایمیل", "تلفن", "شماره تماس", "دبیرخانه",
                     "پشتیبانی", "ارتباط", "چطور تماس", "با کی تماس",
                     "برگزارکننده", "مسئول", "واحد"],
        "intent": "اطلاعات تماس با دبیرخانه نمایشگاه ایران فارما",
    },
    # معرفی نمایشگاه
    {
        "keywords": ["معرفی", "چیه", "چیست", "چی هست", "چیه این",
                     "درباره", "راجع به", "توضیح بده", "بگو",
                     "ایران فارما چیه", "ایران فارما چی"],
        "intent": "معرفی نمایشگاه بین‌المللی ایران‌فارما",
    },
    # امکانات رفاهی
    {
        "keywords": ["امکانات", "رستوران", "غذا", "کافه", "کافی‌شاپ",
                     "نماز", "سرویس بهداشتی", "دستشویی", "wifi", "وای فای",
                     "اینترنت", "ATM", "خودپرداز", "شارژر", "استراحت"],
        "intent": "امکانات رفاهی و خدمات نمایشگاه ایران فارما",
    },
    # برنامه‌های علمی
    {
        "keywords": ["همایش", "کارگاه", "پنل", "سخنرانی", "نشست", "برنامه علمی",
                     "برنامه‌های جانبی", "رویداد", "سمینار", "کنفرانس"],
        "intent": "برنامه‌های علمی و رویدادهای جانبی نمایشگاه ایران فارما",
    },
    # B2B و جلسات تجاری
    {
        "keywords": ["B2B", "جلسه تجاری", "ملاقات تجاری", "شبکه‌سازی",
                     "networking", "همکاری تجاری", "مذاکره", "قرارداد"],
        "intent": "جلسات B2B و شبکه‌سازی در نمایشگاه ایران فارما",
    },
    # دانشجویان
    {
        "keywords": ["دانشجو", "دانشجویی", "دانشگاه", "دانشجویان", "دانشجو میتونه",
                     "دانشجویا", "دانشجو هستم", "دانشجوام"],
        "intent": "حضور دانشجویان در نمایشگاه ایران فارما",
    },
    # استارتاپ
    {
        "keywords": ["استارتاپ", "startup", "شرکت نوپا", "کسب‌وکار نوپا",
                     "ایده", "نوآوری", "دانش‌بنیان"],
        "intent": "حضور استارتاپ‌ها و شرکت‌های دانش‌بنیان در نمایشگاه ایران فارما",
    },
    # صادرات و بین‌الملل
    {
        "keywords": ["صادرات", "بین‌الملل", "خارجی", "بین المللی", "export",
                     "هیئت تجاری", "کشور خارجی", "خارج"],
        "intent": "فرصت‌های صادراتی و بین‌المللی نمایشگاه ایران فارما",
    },
    # سرمایه‌گذاری
    {
        "keywords": ["سرمایه‌گذاری", "سرمایه گذاری", "سرمایه‌گذار", "جذب سرمایه",
                     "funding", "سرمایه"],
        "intent": "فرصت‌های سرمایه‌گذاری در نمایشگاه ایران فارما",
    },
    # نقشه و راهنما
    {
        "keywords": ["نقشه", "راهنما", "دفترچه", "کتاب نمایشگاه", "کاتالوگ",
                     "اپلیکیشن", "اپ", "سایت", "وبسایت", "پلتفرم"],
        "intent": "نقشه و راهنمای نمایشگاه ایران فارما",
    },
    # تبلیغات و اسپانسری
    {
        "keywords": ["تبلیغات", "اسپانسر", "حامی", "آگهی", "بنر", "برندینگ",
                     "تبلیغ", "اسپانسرشیپ"],
        "intent": "تبلیغات و اسپانسرشیپ در نمایشگاه ایران فارما",
    },
]


def detect_intent(user_message: str) -> str | None:
    """
    intent سوال کاربر را تشخیص میدهد.
    specific patterns اول چک میشن تا override نشن.
    """
    msg_norm = normalize_text(user_message)
    best_intent = None
    best_kw_len = 0
    for pattern in INTENT_PATTERNS:
        for kw in pattern["keywords"]:
            kw_norm = normalize_text(kw)
            if kw_norm in msg_norm and len(kw_norm) > best_kw_len:
                best_kw_len = len(kw_norm)
                best_intent = pattern["intent"]
    return best_intent


# ═══════════════════════════════════════════════
#  Query Enricher — rule-based، بدون LLM
# ═══════════════════════════════════════════════
REFERENTIAL_TOKENS = {
    "اونجا","اینجا","اون","این","اونها","اینها","ایشون",
    "همونجا","همینجا","همون","همین","اوناها","بهشون",
    "بیام","برم","برسم","بریم",
}

ENTITY_PATTERNS = [
    (r"(مصل[ای]\s*(?:بزرگ)?\s*(?:امام\s*خمینی)?)", "location"),
    (r"(نمایشگاه\s*(?:بین‌?المللی)?\s*(?:تهران)?)", "location"),
    (r"(سالن\s*\w+)", "location"),
    (r"(تهران)", "location"),
    (r"(\d{1,2}\s*(?:تا|الی)\s*\d{1,2}\s*\w+)", "date"),
    (r"(\d{1,2}\s*\w+\s*(?:ماه)?(?:\s*\d{4})?)", "date"),
    (r"(شرکت\s+[\w\s]+?)(?:\s+(?:در|با|که|را))", "company"),
]


# اگر تاریخچه درباره نمایشگاه بود ولی مکان صریح نداشت، مکان پیش‌فرض inject میشه
EXHIBITION_KEYWORDS = {"نمایشگاه", "ایران فارما", "ایران‌فارما", "iphexpo", "فارما"}
EXHIBITION_LOCATION = "مصلای بزرگ امام خمینی تهران"


def extract_entities_from_history(history: list[ChatMessage], window: int = 4) -> list[str]:
    entities = []
    for msg in history[-window:]:
        text = msg.content
        text_norm = normalize_text(text)
        # اگر پیام درباره نمایشگاه بود، مکان برگزاری رو inject کن
        if any(kw in text_norm for kw in EXHIBITION_KEYWORDS):
            if EXHIBITION_LOCATION not in entities:
                entities.append(EXHIBITION_LOCATION)
        # entity patterns معمولی
        for pattern, _ in ENTITY_PATTERNS:
            for m in re.findall(pattern, text):
                entity = (m.strip() if isinstance(m, str) else m[0].strip())
                if entity and entity not in entities:
                    entities.append(entity)
    return entities


def is_referential(user_message: str) -> bool:
    tokens = set(normalize_text(user_message).split())
    return bool(tokens & REFERENTIAL_TOKENS)


def contextualize_question(user_message: str, history: list[ChatMessage]) -> str:
    """
    دو کار انجام میدهد:
    ۱. Intent detection: سوال عامیانه را به کوئری معنادار تبدیل میکند
    ۲. Entity injection: اگر سوال ارجاعی بود، entity تاریخچه را اضافه میکند
    """
    enriched = user_message

    # ── Entity injection از تاریخچه ──
    if history and is_referential(user_message):
        entities = extract_entities_from_history(history)
        if entities:
            enriched = enriched + " " + entities[0]
            print(f"🔄 Enrich | '{user_message}' → '{enriched}'", flush=True)

    return enriched


# ═══════════════════════════════════════════════
#  جستجوها — همه sync (CPU-bound، بدون I/O)
# ═══════════════════════════════════════════════
def search_examples(user_message: str):
    best, best_score = None, 0.0
    for example in prompt_config.get("examples", []):
        score = similarity(user_message, example.get("input", ""))
        if score > best_score:
            best_score = score
            best = example
    if best and best_score >= 0.90:
        return best.get("response"), best_score
    return None, best_score


def _lang_fields(item: dict, lang: str):
    """
    (question_norm, question, search_text) را برای زبان داده‌شده برمی‌گرداند.
    برای lang="en"، اگر آیتم ترجمه انگلیسی نداشته باشد (question_en_norm یا
    search_text_en خالی) None برمی‌گرداند — یعنی این آیتم در جستجوی انگلیسی
    اصلاً دیده نمی‌شود (به‌جای fallback نادرست به متن فارسی).
    برای lang="fa" (پیش‌فرض) دقیقاً همان فیلدهای قبلی را برمی‌گرداند — رفتار فعلی
    بدون تغییر.
    """
    if lang == "en":
        q_norm = item.get("question_en_norm")
        search_text_en = item.get("search_text_en")
        if not q_norm or not search_text_en:
            return None
        return q_norm, item.get("question_en", ""), search_text_en
    return item["question_norm"], item["question"], item["search_text"]


def _answer_for_lang(item: dict, lang: str) -> str:
    if lang == "en":
        return item.get("answer_en") or ""
    return item["answer"]


def search_exact_knowledge(user_message: str, knowledge_base: list, lang: str = "fa"):
    """
    knowledge_base اکنون به‌صورت صریح پاس داده می‌شود (لیست آیتم‌های همان event) —
    نه global مستقیم، چون درخواست‌های همزمان ممکن است متعلق به eventهای
    متفاوت باشند و نباید global مشترکی را per-request جابه‌جا کرد (race).
    """
    user_norm = normalize_text(user_message)
    best, best_score = None, 0.0
    for item in knowledge_base:
        fields = _lang_fields(item, lang)
        if fields is None:
            continue
        q_norm, _, _ = fields
        if user_norm == q_norm:
            return item, 1.0
        score = similarity(user_norm, q_norm)
        if score > best_score:
            best_score = score
            best = item
    if best and best_score >= 0.92:
        return best, best_score
    return None, best_score


async def search_hybrid_knowledge(
    user_message: str,
    knowledge_base: list,
    index,
    en_indices: list | None,
    top_k: int = 10,
    lang: str = "fa",
) -> list:
    """
    FAISS + keyword + fuzzy.
    embed_text_async یک‌بار await می‌شود — بقیه CPU-bound.
    ترتیب: اول embed (I/O)، بعد FAISS search (CPU).
    knowledge_base/index/en_indices همگی مربوط به یک event خاص هستند (صریحاً
    پاس داده می‌شوند، نه global — همان دلیل search_exact_knowledge).
    lang="en": از index انگلیسی همان event (فقط آیتم‌های دارای ترجمه) و
    فیلدهای انگلیسی استفاده می‌شود؛ آیتم‌های بدون ترجمه اصلاً وارد نتایج نمی‌شوند.
    """
    if index is None or index.ntotal == 0:
        scored = []
        for item in knowledge_base:
            fields = _lang_fields(item, lang)
            if fields is None:
                continue
            _, question, search_text = fields
            scored.append({
                "knowledge": item,
                "score": keyword_score(user_message, search_text) * 0.70
                       + similarity(user_message, question) * 0.30
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ── async embed — این تنها I/O این تابع است ──
    query_vec = np.array([await embed_text_async(user_message)]).astype("float32")
    faiss.normalize_L2(query_vec)
    k = min(top_k, index.ntotal)
    D, I = index.search(query_vec, k)

    scored = []
    for idx, emb_score in zip(I[0], D[0]):
        if idx == -1:
            continue
        kb_idx = en_indices[idx] if en_indices is not None else idx
        if kb_idx >= len(knowledge_base):
            continue
        item = knowledge_base[kb_idx]
        fields = _lang_fields(item, lang)
        if fields is None:
            continue
        _, question, search_text = fields
        score = (
            float(emb_score) * 0.50
            + keyword_score(user_message, search_text) * 0.35
            + similarity(user_message, question) * 0.15
        )
        scored.append({"knowledge": item, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def format_directory_response(user_message: str, raw_answer_json: str, lang: str = "fa") -> str:
    try:
        data     = json.loads(raw_answer_json)
        user_norm = normalize_text(user_message)

        if lang == "en":
            wants_booth_only = (
                any(w in user_norm for w in ["booth", "stand", "hall"])
                and not any(w in user_norm for w in ["address", "phone", "contact", "where", "email"])
            )
            parts = [f"🏢 Company: {data['company']}"]
            if data.get("booth"):
                parts.append(f"📍 Booth No: {data['booth']}")
            if wants_booth_only:
                return "\n".join(parts)
            if data.get("extra"):
                return "\n".join(parts) + "\n\nℹ️ Additional info:\n" + data["extra"]
            return "\n".join(parts)

        wants_booth_only = (
            any(w in user_norm for w in ["غرفه","کدوم سالن","شماره غرفه","کدومه"])
            and not any(w in user_norm for w in ["آدرس","تلفن","شماره تماس","کجاست"])
        )
        parts = [f"🏢 شرکت: {data['company']}"]
        if data.get("booth"):
            parts.append(f"📍 شماره غرفه: {data['booth']}")
        if wants_booth_only:
            return "\n".join(parts)
        if data.get("extra"):
            return "\n".join(parts) + "\n\n📞 اطلاعات تکمیلی:\n" + data["extra"]
        return "\n".join(parts)
    except Exception:
        return raw_answer_json


async def select_best_candidate(user_message: str, candidates: list, lang: str = "fa") -> dict | None:
    """
    Ollama فقط یک عدد برمی‌گرداند.
    async — در حین انتظار Ollama، event loop برای بقیه requestها آزاد است.
    """
    if not candidates:
        return None

    def _cand_question(c):
        k = c["knowledge"]
        if lang == "en":
            return k.get("question_display_en") or k.get("question_en") or k["question"]
        return k.get("question_display") or k["question"]

    options = "".join(f"{i}. {_cand_question(c)}\n" for i, c in enumerate(candidates, 1))

    system_prompt = (
        "You are a strict relevance judge for a pharmaceutical exhibition chatbot. "
        "Given a user question and a list of FAQ options, output the number of the option "
        "that DIRECTLY and SPECIFICALLY answers the user's question. "
        "CRITICAL: If the user is asking about a general topic or about a specific "
        "company/person NOT mentioned in the options, output 0. "
        "Output ONLY the single digit number, absolutely no other text."
    )
    user_prompt = f"User Question: {user_message}\n\nOptions:\n{options}\n\nAnswer (0 if none match):"

    raw = await call_ollama_async(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        num_predict=3,
        temperature=0,
    )

    print(f"🔍 Ollama raw response: {repr(raw)}", flush=True)
    print(f"🔍 Options sent:\n{options}", flush=True)
    if raw:
        m = re.search(r"\d+", raw)
        if m:
            idx = int(m.group())
            if 1 <= idx <= len(candidates):
                print(f"🤖 Ollama selected #{idx}", flush=True)
                return candidates[idx - 1]
            if idx == 0:
                print("🤖 Ollama: no match", flush=True)
                return None

    # timeout یا خطا — بهترین FAISS score
    return candidates[0] if candidates else None


# ═══════════════════════════════════════════════
#  endpoint اصلی — async
# ═══════════════════════════════════════════════
@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    event_id = req.event_id
    user_uuid = req.user_uuid
    if not user_message:
        return {"answer": get_fallback_message(event_id, req.lang), "source": "empty"}

    # ── بررسی سوالات meta درباره تاریخچه (مستقل از event، فقط از history استفاده می‌کند) ──
    meta_keywords = ["سوال قبلی", "قبلاً چی گفتم", "قبلا چی گفتم", "آخرین سوالم",
                     "چی پرسیدم", "چی گفتم", "سوالم چی بود", "قبلی چی بود"]
    msg_norm_meta = normalize_text(user_message)
    if any(normalize_text(kw) in msg_norm_meta for kw in meta_keywords):
        user_questions = [m.content for m in req.history if m.role == "user"]
        if user_questions:
            last_q = user_questions[-1]
            ans = f"آخرین سوال شما این بود: «{last_q}»"
        else:
            ans = "تاریخچه‌ای از سوالات شما وجود ندارد."
        await log_chat_interaction(user_message, ans, "meta", 1.0, event_id=event_id, user_uuid=user_uuid)
        return {"answer": ans, "source": "meta"}

    # ۱. Query Enricher (sync — CPU)
    search_query = contextualize_question(user_message, req.history)

    # ۲. Example search (sync — CPU، مستقل از event، از prompt_config می‌آید)
    example_answer, _ = search_examples(search_query)
    if example_answer:
        await log_chat_interaction(user_message, example_answer, "example", 1.0, search_query, event_id=event_id, user_uuid=user_uuid)
        return {"answer": example_answer, "source": "example"}

    # ── این event هنوز هیچ KB‌ای ندارد (هرگز rebuild نشده یا واقعاً خالی است) ──
    # مستقیم به fallback همان event برو — نه خطا، نه fallthrough به event دیگر.
    knowledge_base = KNOWLEDGE_BASE.get(event_id)
    if not knowledge_base:
        fallback = get_fallback_message(event_id, req.lang)
        await log_chat_interaction(user_message, fallback, "fallback_no_kb", 0.0, event_id=event_id, user_uuid=user_uuid)
        return {"answer": fallback, "source": "fallback_no_kb"}

    # ۳. Exact / fuzzy search (sync — CPU)
    exact_item, exact_score = search_exact_knowledge(search_query, knowledge_base, lang=req.lang)
    if exact_item:
        ans = _answer_for_lang(exact_item, req.lang)
        if exact_item.get("is_directory"):
            ans = format_directory_response(user_message, ans, lang=req.lang)
        matched_q = exact_item.get("question_en") if req.lang == "en" else exact_item["question"]
        await log_chat_interaction(user_message, ans, "exact", exact_score, matched_q, event_id=event_id, user_uuid=user_uuid)
        return {"answer": ans, "source": "exact"}

    # ۴-۷ همه به Ollama نیاز دارند (embedding در search_hybrid_knowledge،
    # انتخاب در select_best_candidate) — پس زیر سقف همزمانی محلی قرار می‌گیرند.
    # اگر ظرف CHAT_QUEUE_WAIT_SECONDS نوبت آزاد نشد، پیام «شلوغ» برمی‌گردد
    # (بدون رسیدن به Ollama) تا بار روی VPS در زمان قطعی GPU کنترل‌شده بماند.
    try:
        await asyncio.wait_for(_chat_llm_semaphore.acquire(), timeout=CHAT_QUEUE_WAIT_SECONDS)
    except asyncio.TimeoutError:
        busy = CAPACITY_BUSY_MESSAGE.get(req.lang, CAPACITY_BUSY_MESSAGE["fa"])
        await log_chat_interaction(user_message, busy, "capacity_limited", 0.0, event_id=event_id, user_uuid=user_uuid)
        return {"answer": busy, "source": "capacity_limited"}

    try:
        # ۴. FAISS + hybrid (async — شامل embed I/O)
        index = faiss_index_en.get(event_id) if req.lang == "en" else faiss_index.get(event_id)
        en_indices = KB_EN_INDICES.get(event_id) if req.lang == "en" else None
        candidates = await search_hybrid_knowledge(
            search_query, knowledge_base, index, en_indices, top_k=10, lang=req.lang
        )
        if not candidates:
            fallback = get_fallback_message(event_id, req.lang)
            await log_chat_interaction(user_message, fallback, "fallback", 0.0, event_id=event_id, user_uuid=user_uuid)
            return {"answer": fallback, "source": "fallback"}

        print(f"📊 scores: {[round(c['score'],3) for c in candidates]}", flush=True)

        # ۵. Threshold filter
        MIN_SCORE  = 0.50
        candidates = [c for c in candidates if c["score"] >= MIN_SCORE]
        if not candidates:
            fallback = get_fallback_message(event_id, req.lang)
            await log_chat_interaction(user_message, fallback, "fallback_threshold", 0.0, event_id=event_id, user_uuid=user_uuid)
            return {"answer": fallback, "source": "fallback"}

        # ۶. Ollama انتخاب (async — I/O)
        selected = await select_best_candidate(search_query, candidates[:5], lang=req.lang)
        if not selected:
            fallback = get_fallback_message(event_id, req.lang)
            await log_chat_interaction(user_message, fallback, "fallback", 0.0, event_id=event_id, user_uuid=user_uuid)
            return {"answer": fallback, "source": "fallback"}

        # ۷. فرمت و برگشت
        ans = _answer_for_lang(selected["knowledge"], req.lang)
        if selected["knowledge"].get("is_directory"):
            ans = format_directory_response(user_message, ans, lang=req.lang)

        matched_q = selected["knowledge"].get("question_en") if req.lang == "en" else selected["knowledge"]["question"]
        await log_chat_interaction(user_message, ans, "rag", selected["score"], matched_q, event_id=event_id, user_uuid=user_uuid)
        return {"answer": ans, "source": "rag"}
    finally:
        _chat_llm_semaphore.release()


# ═══════════════════════════════════════════════
#  endpoint لاگ — برای پنل مدیریت
# ═══════════════════════════════════════════════
@app.get("/logs")
async def get_logs(source: str = None, event_id: int = None, limit: int = 500, _: None = Depends(verify_admin_key)):
    logs = []
    if not LOG_FILE_PATH.exists():
        return {"logs": []}
    try:
        with open(LOG_FILE_PATH, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if source and row.get("Source", "").lower() != source.lower():
                    continue
                # لاگ‌های قدیمی‌تر از این migration ستون Event_Id ندارند — event_id=None
                # برایشان می‌ماند و فیلتر event_id روی آن‌ها اعمال نمی‌شود (تا گم نشوند).
                row_event_id = row.get("Event_Id") or None
                if event_id is not None and row_event_id is not None and str(row_event_id) != str(event_id):
                    continue
                logs.append({
                    "timestamp":       row.get("Timestamp", ""),
                    "user_message":    row.get("User_Message", ""),
                    "bot_answer":      row.get("Bot_Answer", ""),
                    "source":          row.get("Source", ""),
                    "score":           row.get("Score", "0"),
                    "matched_question": row.get("Matched_Question", ""),
                    "event_id":        row_event_id,
                    # Blank for rows written before this column existed, and
                    # for guest/unauthenticated chats -- both expected, not
                    # errors. iph-apn resolves this to a name/mobile at
                    # display time via a live app_users join, never stored
                    # here.
                    "user_uuid":       row.get("User_Uuid") or None,
                })
    except Exception as e:
        return {"logs": [], "error": str(e)}
    return {"logs": logs[-limit:]}


@app.delete("/logs")
async def delete_logs(event_id: int = Query(...), _: None = Depends(verify_admin_key)):
    """
    فقط ردیف‌هایی که Event_Id آن‌ها دقیقاً برابر event_id است حذف می‌شوند —
    ردیف‌های event دیگر و ردیف‌های قدیمی بدون Event_Id (مطابق همان استثنای
    GET /logs، تا گم نشوند) دست‌نخورده می‌مانند. نوشتن با temp-file + os.replace
    اتمیک است تا اگر پردازه وسط نوشتن kill شود، فایل اصلی نصفه‌نویسی‌شده باقی
    نماند. زیر همان _log_lock که log_chat_interaction استفاده می‌کند، تا با
    نوشتن‌های همزمان لاگ جدید race نکند.
    """
    async with _log_lock:
        if not LOG_FILE_PATH.exists():
            return {"deleted": 0}

        deleted = 0
        kept_rows = []
        try:
            with open(LOG_FILE_PATH, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or LOG_HEADER
                for row in reader:
                    row_event_id = row.get("Event_Id") or None
                    if row_event_id is not None and str(row_event_id) == str(event_id):
                        deleted += 1
                        continue
                    kept_rows.append(row)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read logs: {e}")

        tmp_path = LOG_FILE_PATH.with_suffix(".csv.tmp")
        try:
            with open(tmp_path, mode="w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept_rows)
            os.replace(tmp_path, LOG_FILE_PATH)
        except Exception as e:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to write logs: {e}")

    return {"deleted": deleted}


# ═══════════════════════════════════════════════
#  Admin CRUD — مدیریت FAQ (Postgres)
#  همه‌ی endpointها با هدر X-Admin-Key محافظت می‌شوند.
#  بعد از هر تغییر، embeddings/FAISS بلافاصله و خودکار rebuild می‌شود.
# ═══════════════════════════════════════════════
def _faq_row_to_dict(row) -> dict:
    return {
        "id": row[0], "category": row[1], "question": row[2], "answer": row[3],
        "question_en": row[4] if len(row) > 4 else None,
        "answer_en": row[5] if len(row) > 5 else None,
        "synced_to_primary": row[6] if len(row) > 6 else None,
    }


def _generate_faq_id(existing_ids: set) -> str:
    while True:
        candidate = "".join(secrets.choice(FAQ_ID_ALPHABET) for _ in range(8))
        if candidate not in existing_ids:
            return candidate


@app.get("/admin/faq")
async def admin_list_faq(event_id: int = Query(...), _: None = Depends(verify_admin_key)):
    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, category, question, answer, question_en, answer_en, synced_to_primary "
                "FROM faq WHERE event_id = %s ORDER BY created_at",
                (event_id,),
            )
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    return [_faq_row_to_dict(r) for r in rows]


@app.post("/admin/faq")
async def admin_create_faq(payload: FAQCreate, event_id: int = Query(...), _: None = Depends(verify_admin_key)):
    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM faq")
            existing_ids = {r[0] for r in cur.fetchall()}
            if payload.id is not None and payload.id.strip():
                new_id = payload.id
            else:
                new_id = _generate_faq_id(existing_ids)
            cur.execute(
                """
                INSERT INTO faq (id, category, question, answer, question_en, answer_en, synced_to_primary, event_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, category, question, answer, question_en, answer_en, synced_to_primary
                """,
                (new_id, payload.category, payload.question, payload.answer,
                 payload.question_en, payload.answer_en, payload.synced_to_primary, event_id),
            )
            row = cur.fetchone()
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    await rebuild_knowledge_base(event_id)
    return _faq_row_to_dict(row)


@app.put("/admin/faq/{faq_id}")
async def admin_update_faq(faq_id: str, payload: FAQUpdate, event_id: int = Query(...), _: None = Depends(verify_admin_key)):
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    set_clauses = [f"{field} = %s" for field in updates] + ["updated_at = NOW()"]
    # event_id در WHERE به‌عنوان یک guard دفاعی اضافه شده — id‌های FAQ به‌صورت
    # سراسری یکتا هستند (رشته‌ی تصادفی ۸ کاراکتری)، پس این فقط از یک ادمین با
    # event فعلی متفاوت جلوگیری می‌کند که با یک id شناخته‌شده/leak-شده ردیف
    # event دیگری را ویرایش کند — نه یک نیاز فنی برای پیدا کردن ردیف.
    values = list(updates.values()) + [faq_id, event_id]

    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE faq SET {', '.join(set_clauses)} WHERE id = %s AND event_id = %s "
                f"RETURNING id, category, question, answer, question_en, answer_en, synced_to_primary",
                values,
            )
            row = cur.fetchone()
        if row is None:
            conn.rollback()
            raise HTTPException(status_code=404, detail="FAQ id not found")
        conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    await rebuild_knowledge_base(event_id)
    return _faq_row_to_dict(row)


@app.delete("/admin/faq/{faq_id}")
async def admin_delete_faq(faq_id: str, event_id: int = Query(...), _: None = Depends(verify_admin_key)):
    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM faq WHERE id = %s AND event_id = %s", (faq_id, event_id))
            deleted = cur.rowcount
        if deleted == 0:
            conn.rollback()
            raise HTTPException(status_code=404, detail="FAQ id not found")
        conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    await rebuild_knowledge_base(event_id)
    return {"deleted": True}


# ═══════════════════════════════════════════════
#  Admin CRUD — تنظیمات سراسری bot (Postgres, bot_settings)
#  همان محافظت X-Admin-Key. بعد از هر PUT، BOT_SETTINGS بلافاصله از دیتابیس
#  دوباره خوانده می‌شود — یعنی تغییر بدون نیاز به ری‌استارت روی درخواست بعدی
#  /chat اعمال می‌شود.
# ═══════════════════════════════════════════════
@app.get("/admin/bot-settings")
async def admin_get_bot_settings(event_id: int = Query(...), _: None = Depends(verify_admin_key)):
    return BOT_SETTINGS.get(event_id, {})


@app.put("/admin/bot-settings/{key}")
async def admin_update_bot_setting(key: str, payload: BotSettingUpdate, event_id: int = Query(...), _: None = Depends(verify_admin_key)):
    global BOT_SETTINGS

    if payload.value_fa is None and payload.value_en is None:
        raise HTTPException(status_code=400, detail="No fields to update")

    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor() as cur:
            # فیلدی که ارسال نشده (None) دست‌نخورده باقی می‌ماند — COALESCE با
            # مقدار قبلی همان ستون در صورت conflict (partial update روی upsert).
            cur.execute(
                """
                INSERT INTO bot_settings (event_id, key, value_fa, value_en, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (event_id, key) DO UPDATE SET
                    value_fa   = COALESCE(EXCLUDED.value_fa, bot_settings.value_fa),
                    value_en   = COALESCE(EXCLUDED.value_en, bot_settings.value_en),
                    updated_at = NOW()
                """,
                (event_id, key, payload.value_fa, payload.value_en),
            )
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    BOT_SETTINGS = load_bot_settings()
    return BOT_SETTINGS.get(event_id, {})


# ═══════════════════════════════════════════════
#  Admin Sync — مدیریت شرکت‌ها/غرفه‌داران (Postgres)
#  آینه‌ای از دیتای شرکت‌ها که پنل ادمین (روی سرور دیگر) push می‌کند.
#  همان محافظت X-Admin-Key؛ بعد از هر sync، KNOWLEDGE_BASE/FAISS خودکار rebuild می‌شود.
# ═══════════════════════════════════════════════
COMPANY_FIELDS = [
    "id", "brand_name_fa", "brand_name_en", "legal_name_fa", "legal_name_en", "logo",
    "website", "description_fa", "description_en", "slug", "phones", "emails",
    "address_fa", "address_en", "industry_id", "hall_name", "booth_no", "is_sponsor",
    "sponsor_level", "booth_uuid", "booth_xp", "is_manual", "linked_mission_id",
    "linked_badge_id", "repeatable_scan", "repeatable_scan_hours", "repeatable_start_hour",
]
COMPANY_JSON_FIELDS = {"logo", "phones", "emails"}
COMPANY_BOOLEAN_FIELDS = {"is_sponsor", "is_manual", "repeatable_scan"}  # DEFAULT false در schema

# rasayesh_event_id (شرکت‌ها به کدام رویداد Rasayesh تعلق دارند) جدا از local
# event_id (کدام رویداد محلی — IranPharma=1، Iran Cosmetica=2 و ...) — دو
# مفهوم متفاوت که قبلاً هر دو "event_id" نامیده می‌شدند (سردرگمی که در این
# migration رفع شد، مشابه Phase 1 اپ اصلی).
_COMPANY_ALL_COLUMNS = COMPANY_FIELDS + ["rasayesh_event_id", "event_id"]
_COMPANY_INSERT_SQL = (
    f"INSERT INTO companies ({', '.join(_COMPANY_ALL_COLUMNS)}) "
    f"VALUES ({', '.join(['%s'] * len(_COMPANY_ALL_COLUMNS))}) "
    f"ON CONFLICT (id) DO UPDATE SET "
    + ", ".join(f"{col} = EXCLUDED.{col}" for col in COMPANY_FIELDS if col != "id")
    + ", rasayesh_event_id = EXCLUDED.rasayesh_event_id, event_id = EXCLUDED.event_id, synced_at = NOW()"
)


@app.get("/admin/companies")
async def admin_list_companies(event_id: int = Query(...), _: None = Depends(verify_admin_key)):
    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM companies WHERE event_id = %s ORDER BY brand_name_fa", (event_id,))
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    return [dict(r) for r in rows]


@app.post("/admin/companies/sync")
async def admin_sync_companies(payload: dict = Body(...), _: None = Depends(verify_admin_key)):
    # event_id: local event id این چت‌بات (iph-apn هر بار صریحاً می‌فرستد).
    # rasayesh_event_id: id همان رویداد در سیستم خارجی Rasayesh — برای نمایش/
    # ردیابی نگه داشته می‌شود، در تشخیص "این شرکت مال کدام local event است"
    # نقشی ندارد.
    event_id = payload.get("event_id")
    rasayesh_event_id = payload.get("rasayesh_event_id")
    companies = payload.get("companies")

    if not isinstance(event_id, int) or isinstance(event_id, bool):
        raise HTTPException(status_code=400, detail="'event_id' must be an integer")
    if not isinstance(rasayesh_event_id, int) or isinstance(rasayesh_event_id, bool):
        raise HTTPException(status_code=400, detail="'rasayesh_event_id' must be an integer")
    if not isinstance(companies, list):
        raise HTTPException(status_code=400, detail="'companies' must be a list")
    for i, c in enumerate(companies):
        if not isinstance(c, dict) or "id" not in c:
            raise HTTPException(status_code=400, detail=f"companies[{i}] must be an object with an 'id' field")

    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor() as cur:
            # فقط شرکت‌های همین local event پاک/جایگزین می‌شوند — sync یک event
            # هرگز شرکت‌های eventهای دیگر را حذف نمی‌کند (قبلاً با != کل جدول
            # به‌عنوان single-tenant پاک می‌شد، که با چند event نادرست است).
            cur.execute("DELETE FROM companies WHERE event_id = %s", (event_id,))
            for c in companies:
                values = []
                for field in COMPANY_FIELDS:
                    value = c.get(field)
                    if field in COMPANY_JSON_FIELDS and value is not None:
                        value = Json(value)
                    elif field in COMPANY_BOOLEAN_FIELDS and value is None:
                        value = False
                    values.append(value)
                values.append(rasayesh_event_id)
                values.append(event_id)
                cur.execute(_COMPANY_INSERT_SQL, values)
        conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    await rebuild_knowledge_base(event_id)
    return {"synced": len(companies), "event_id": event_id, "rasayesh_event_id": rasayesh_event_id}


# ═══════════════════════════════════════════════
#  Admin — پنل‌ها/کارگاه‌ها (panels)
#  همان محافظت X-Admin-Key؛ بعد از هر sync، KNOWLEDGE_BASE/FAISS خودکار rebuild می‌شود.
#  kind (PANEL/WORKSHOP) فقط یک ستون محتوایی است، نه یک منبع جدا.
# ═══════════════════════════════════════════════
PANEL_FIELDS = [
    "id", "title_fa", "title_en", "description_fa", "description_en",
    "hall_fa", "hall_en", "starts_at", "ends_at", "capacity", "kind",
    "thumbnail", "speakers",
]
PANEL_JSON_FIELDS = {"thumbnail", "speakers"}

_PANEL_ALL_COLUMNS = PANEL_FIELDS + ["rasayesh_event_id", "event_id"]
_PANEL_INSERT_SQL = (
    f"INSERT INTO panels ({', '.join(_PANEL_ALL_COLUMNS)}) "
    f"VALUES ({', '.join(['%s'] * len(_PANEL_ALL_COLUMNS))}) "
    f"ON CONFLICT (id) DO UPDATE SET "
    + ", ".join(f"{col} = EXCLUDED.{col}" for col in PANEL_FIELDS if col != "id")
    + ", rasayesh_event_id = EXCLUDED.rasayesh_event_id, event_id = EXCLUDED.event_id, synced_at = NOW()"
)


@app.get("/admin/panels")
async def admin_list_panels(event_id: int = Query(...), _: None = Depends(verify_admin_key)):
    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM panels WHERE event_id = %s ORDER BY starts_at NULLS LAST", (event_id,))
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    return [dict(r) for r in rows]


@app.post("/admin/panels/sync")
async def admin_sync_panels(payload: dict = Body(...), _: None = Depends(verify_admin_key)):
    # همان قرارداد companies: event_id لوکال این چت‌بات، rasayesh_event_id
    # فقط برای نمایش/ردیابی.
    event_id = payload.get("event_id")
    rasayesh_event_id = payload.get("rasayesh_event_id")
    panels = payload.get("panels")

    if not isinstance(event_id, int) or isinstance(event_id, bool):
        raise HTTPException(status_code=400, detail="'event_id' must be an integer")
    if not isinstance(rasayesh_event_id, int) or isinstance(rasayesh_event_id, bool):
        raise HTTPException(status_code=400, detail="'rasayesh_event_id' must be an integer")
    if not isinstance(panels, list):
        raise HTTPException(status_code=400, detail="'panels' must be a list")
    for i, p in enumerate(panels):
        if not isinstance(p, dict) or "id" not in p:
            raise HTTPException(status_code=400, detail=f"panels[{i}] must be an object with an 'id' field")

    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor() as cur:
            # فقط پنل/کارگاه‌های همین local event پاک/جایگزین می‌شوند (مثل companies).
            cur.execute("DELETE FROM panels WHERE event_id = %s", (event_id,))
            for p in panels:
                values = []
                for field in PANEL_FIELDS:
                    value = p.get(field)
                    if field in PANEL_JSON_FIELDS and value is not None:
                        value = Json(value)
                    values.append(value)
                values.append(rasayesh_event_id)
                values.append(event_id)
                cur.execute(_PANEL_INSERT_SQL, values)
        conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    await rebuild_knowledge_base(event_id)
    return {"synced": len(panels), "event_id": event_id, "rasayesh_event_id": rasayesh_event_id}
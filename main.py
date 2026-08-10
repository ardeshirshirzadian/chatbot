import os
import json
import csv
import re
import secrets
import string
import asyncio
import logging
import time
from dataclasses import dataclass
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
from fastapi import FastAPI, Header, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# ── لاگ ساختاریافته (timestamp + level) — برای دیدن خطاهای Ollama که قبلاً بی‌صدا فرو می‌افتادند ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("iranpharma_chat")

# هر چند دقیقه یک‌بار مدل چت را با یک درخواست حداقلی بیدار نگه می‌دارد —
# شبکه ایمنی مستقل از OLLAMA_KEEP_ALIVE، برای مواقعی که کانتینر Ollama
# مستقل از این سرویس ری‌استارت شود و مدل از حافظه GPU خارج شود.
OLLAMA_REWARM_INTERVAL_SECONDS = 240

# ── صف پردازش با ظرفیت محدود — جایگزین رد سخت --limit-concurrency ──
# به‌جای رد فوری درخواست ۶۱ام، اگر همه‌ی PROCESSING_SLOTS اشغال باشند، درخواست
# در صف (تا سقف MAX_QUEUE_DEPTH) منتظر می‌ماند تا یک پردازش پیدا کند، به‌جای رد فوری.
#
# مقادیر بر اساس تست بار امروز: تا ۱۰۰ درخواست همزمان صفر خطای واقعی، فقط
# افزایش latency (میانگین ۸ تا ۱۶ ثانیه). بنابراین:
#   - PROCESSING_SLOTS=60 همان سقف قبلی را حفظ می‌کند (منطقه‌ی شناخته‌شده و امن از نظر latency).
#   - MAX_QUEUE_DEPTH=40 یعنی PROCESSING_SLOTS + MAX_QUEUE_DEPTH = 100 — دقیقاً همان
#     سقفی که در تست بار واقعاً معتبرسنجی شد، نه فراتر از آن.
#   - MAX_WAIT_SECONDS=30 (نه ۱۸ پیشنهادی اولیه): یک job صف‌شده ممکن است تا ~16s برای
#     آزاد شدن یک slot صبر کند و سپس خودش تا ~16s پردازش شود (~32s بدترین حالت واقع‌گرایانه).
#     18s قبل از تمام‌شدن اکثر jobهای صف‌شده، "busy" کاذب برمی‌گرداند؛ 30s این حاشیه را پوشش می‌دهد.
PROCESSING_SLOTS = 60
MAX_QUEUE_DEPTH = 40
MAX_WAIT_SECONDS = 30
QUEUE_RESULT_TTL_SECONDS = 120  # مدت نگهداری نتیجه‌ی تکمیل‌شده در حافظه قبل از پاک‌سازی
AVG_PIPELINE_SECONDS = 12       # میانگین تقریبی مشاهده‌شده (۸ تا ۱۶ ثانیه) — فقط برای تخمین زمان انتظار

BUSY_MESSAGE_FA = "الان شلوغه، لطفاً بعداً امتحان کنید."

DEFAULT_FALLBACK = "این سؤال خارج از حوزه نمایشگاه ایران‌فارما است یا اطلاعات آن در پایگاه دانش ثبت نشده است."

# ── یک httpx.AsyncClient مشترک برای کل اپ ──────────────────────────
# keep-alive + connection pool — در روزهای نمایشگاه فشار کمتری روی Ollama
_http: httpx.AsyncClient = None
_rewarm_task: asyncio.Task = None

# ── صف /chat و worker pool — ساخته می‌شوند در lifespan startup ──
_job_queue: asyncio.Queue = None
_slot_semaphore: asyncio.Semaphore = None
_queue_worker_tasks: list = []
_queue_results: dict = {}   # queue_id → {"state": "queued"|"done", ...}
_next_seq = 0                # شماره‌ی افزایشی هر job که وارد صف می‌شود
_dequeued_count = 0          # تعداد jobهایی که تا الان از صف خارج شده‌اند (برای محاسبه‌ی position)


@dataclass
class _QueueJob:
    queue_id: str
    req: "ChatRequest"
    seq: int
    enqueued_at: float

# ═══════════════════════════════════════════════
#  حالت‌های Global
# ═══════════════════════════════════════════════
KNOWLEDGE_BASE  = []
FALLBACK        = DEFAULT_FALLBACK
prompt_config   = {}
faiss_index     = None
faiss_index_en  = None   # index موازی روی question_en_norm — فقط آیتم‌هایی که ترجمه انگلیسی دارند
en_index_to_kb  = []      # نگاشت موقعیت در faiss_index_en → اندیس در KNOWLEDGE_BASE
embedding_dimension = None

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


async def call_ollama_async(messages: list, num_predict: int = 3, temperature: float = 0, context: str = "") -> str | None:
    """
    یک فراخوانی async به Ollama chat API.
    stream=False — فقط یک عدد برمی‌گرداند (انتخاب کاندیدا).
    timeout=30s — اگر مدل کند بود graceful timeout.
    context: برچسب محل فراخوانی (مثلاً "select_candidate"، "warmup") — برای تشخیص‌پذیری در لاگ خطا.
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
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except httpx.TimeoutException:
        logger.warning(f"Ollama timeout (context={context or 'unknown'})")
    except Exception as e:
        logger.warning(f"Ollama error (context={context or 'unknown'}): {type(e).__name__}: {e}")
    return None


async def warmup_ollama_model():
    """
    یک درخواست حداقلی به مدل chat می‌فرستد تا در حافظه GPU لود شود —
    هم در startup (به‌جای اینکه اولین کاربر واقعی منتظر cold start بماند)
    و هم به‌صورت دوره‌ای از periodic_ollama_rewarm.
    """
    try:
        return await call_ollama_async(
            messages=[{"role": "user", "content": "hi"}],
            num_predict=1,
            temperature=0,
            context="warmup",
        )
    except Exception as e:
        logger.warning(f"Ollama warmup failed: {type(e).__name__}: {e}")
        return None


async def periodic_ollama_rewarm():
    """
    شبکه ایمنی مستقل از OLLAMA_KEEP_ALIVE: هر چند دقیقه یک‌بار مدل چت را
    با یک درخواست حداقلی (num_predict=1) بیدار نگه می‌دارد. اگر کانتینر
    Ollama مستقل از این سرویس ری‌استارت شده و مدل از GPU خارج شده باشد،
    این تسک ظرف چند دقیقه دوباره لودش می‌کند — نه یک کاربر واقعی.
    """
    while True:
        await asyncio.sleep(OLLAMA_REWARM_INTERVAL_SECONDS)
        try:
            result = await warmup_ollama_model()
            if result is None:
                logger.warning("Periodic Ollama re-warm ping got no response")
        except Exception as e:
            logger.warning(f"Periodic Ollama re-warm ping failed: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════
#  لاگ async — بدون race condition
# ═══════════════════════════════════════════════
async def log_chat_interaction(user_msg: str, bot_ans: str, source: str, score: float, matched_q: str = ""):
    async with _log_lock:
        try:
            file_exists = LOG_FILE_PATH.exists()
            LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "User_Message", "Bot_Answer", "Source", "Score", "Matched_Question"])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user_msg, bot_ans, source, score, matched_q
                ])
        except Exception:
            pass


# ═══════════════════════════════════════════════
#  بارگذاری Knowledge Base (sync — فقط startup / rebuild)
# ═══════════════════════════════════════════════
def load_faq_from_postgres():
    """FAQ items از جدول Postgres `faq` — جایگزین knowledge/faq.csv."""
    faq_items = []
    try:
        conn = get_faq_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, category, question, answer, question_en, answer_en "
                    "FROM faq ORDER BY created_at"
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        print(f"Error loading FAQ from Postgres: {e}", flush=True)
        return faq_items

    for row_id, category, question, answer, question_en, answer_en in rows:
        question = (question or "").strip()
        answer   = (answer or "").strip()
        category = (category or "عمومی").strip()
        question_en = (question_en or "").strip()
        answer_en   = (answer_en or "").strip()
        if question and answer:
            # فیلدهای انگلیسی فقط وقتی هر دو question_en و answer_en موجودند ست می‌شوند —
            # در غیر این صورت این آیتم در جستجوی انگلیسی نامرئی می‌ماند (به‌جای fallback به فارسی)
            has_en = bool(question_en and answer_en)
            faq_items.append({
                "id": row_id,
                "category": category,
                "question": question,
                "question_norm": normalize_text(question),
                "answer": answer,
                "search_text": build_search_text(question, answer, category, "postgres:faq"),
                "source_file": "postgres:faq",
                "is_directory": False,
                "question_en": question_en if has_en else None,
                "answer_en": answer_en if has_en else None,
                "question_en_norm": normalize_text(question_en) if has_en else None,
                "search_text_en": build_search_text(question_en, answer_en, category, "postgres:faq") if has_en else None,
            })

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


def load_companies_from_postgres():
    """آیتم‌های دایرکتوری شرکت‌ها از جدول Postgres `companies` — جایگزین knowledge/companies.csv."""
    company_items = []
    try:
        conn = get_faq_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, brand_name_fa, brand_name_en, hall_name, booth_no, website,
                           phones, emails, address_fa, description_en, address_en
                    FROM companies
                    """
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        print(f"Error loading companies from Postgres: {e}", flush=True)
        return company_items

    for (row_id, brand_name_fa, brand_name_en, hall_name, booth_no, website,
         phones, emails, address_fa, description_en, address_en) in rows:
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

        # ── نسخه انگلیسی (اختیاری) — فقط اگر brand_name_en موجود باشد ساخته می‌شود.
        # فیلدهای غیر فارسی (وب‌سایت/تلفن/سالن/غرفه) در هر دو حالت قابل استفاده‌اند،
        # اما پروز فارسی (description_fa/address_fa) هرگز وارد پاسخ انگلیسی نمی‌شود.
        company_name_en = (brand_name_en or "").strip()
        question_en = answer_en = question_en_norm = search_text_en = None
        if company_name_en:
            question_en = (
                f"What is the booth and contact information for {company_name_en}? "
                f"Is {company_name_en} present at the exhibition? Where is {company_name_en}'s booth? "
                f"{company_name_en} booth number"
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
            description_en = (description_en or "").strip()
            if description_en:
                extra_parts_en.append(f"• About: {description_en}")
            address_en = (address_en or "").strip()
            if address_en:
                extra_parts_en.append(f"• Address: {address_en}")

            answer_en = json.dumps(
                {"company": company_name_en, "booth": booth_no, "extra": "\n".join(extra_parts_en)},
                ensure_ascii=False
            )
            question_en_norm = normalize_text(question_en)
            search_text_en = build_search_text(question_en, answer_en, category, "postgres:companies")

        company_items.append({
            "id": f"company_{row_id}",
            "category": category,
            "question": question,
            "question_norm": normalize_text(question),
            "answer": answer,
            "search_text": build_search_text(question, answer, category, "postgres:companies"),
            "source_file": "postgres:companies",
            "is_directory": True,
            "question_en": question_en,
            "answer_en": answer_en,
            "question_en_norm": question_en_norm,
            "search_text_en": search_text_en,
        })

    return company_items


def load_all_knowledge_bases():
    knowledge_list = load_faq_from_postgres()
    knowledge_list.extend(load_companies_from_postgres())

    if not KNOWLEDGE_DIR.exists():
        return knowledge_list

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
# ═══════════════════════════════════════════════
async def rebuild_knowledge_base():
    """
    KNOWLEDGE_BASE را از Postgres (FAQ) + CSVها (مثل companies.csv) دوباره می‌سازد،
    فقط آیتم‌های جدید/تغییریافته را embed می‌کند (با استفاده از cache موجود)
    و FAISS index را از نو می‌سازد. global هایی که /chat استفاده می‌کند به‌روز می‌شوند.
    """
    global KNOWLEDGE_BASE, faiss_index, faiss_index_en, en_index_to_kb, embedding_dimension

    KNOWLEDGE_BASE = load_all_knowledge_bases()
    print(f"✅ Knowledge base loaded: {len(KNOWLEDGE_BASE)} items", flush=True)

    # ── بارگذاری cache ──
    cached_embeddings = {}
    if EMBEDDINGS_CACHE_PATH.exists():
        try:
            with open(EMBEDDINGS_CACHE_PATH, "r", encoding="utf-8") as f:
                cached_embeddings = json.load(f)
        except Exception:
            pass

    # ── embedding موازی برای آیتم‌های جدید ──
    # آیتم‌هایی که cache ندارند همزمان embed می‌شوند (asyncio.gather)
    # ترتیب KNOWLEDGE_BASE حفظ می‌شود
    keys_to_embed = [
        (i, item["question_norm"])
        for i, item in enumerate(KNOWLEDGE_BASE)
        if item["question_norm"] not in cached_embeddings
    ]

    # آیتم‌هایی که ترجمه انگلیسی دارند هم باید embed شوند — برای faiss_index_en
    keys_to_embed_en = [
        (i, item["question_en_norm"])
        for i, item in enumerate(KNOWLEDGE_BASE)
        if item.get("question_en_norm") and item["question_en_norm"] not in cached_embeddings
    ]

    all_keys_to_embed = keys_to_embed + keys_to_embed_en

    if all_keys_to_embed:
        print(f"🔄 Embedding {len(all_keys_to_embed)} new items (parallel)...", flush=True)
        # batch ها را گروه‌بندی کن — ۱۰ تایی تا Ollama اشباع نشود
        BATCH = 10
        for batch_start in range(0, len(all_keys_to_embed), BATCH):
            batch = all_keys_to_embed[batch_start: batch_start + BATCH]
            results = await asyncio.gather(
                *[embed_text_async(key) for _, key in batch],
                return_exceptions=True
            )
            for (idx, key), result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"⚠️  Embedding error for item {idx}: {result}", flush=True)
                else:
                    cached_embeddings[key] = result

    # ذخیره cache
    try:
        EMBEDDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EMBEDDINGS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cached_embeddings, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # ── ساخت FAISS index فارسی (بدون تغییر) ──
    embedding_list = [
        cached_embeddings[item["question_norm"]]
        for item in KNOWLEDGE_BASE
        if item["question_norm"] in cached_embeddings
    ]

    if embedding_list:
        emb_np = np.array(embedding_list).astype("float32")
        embedding_dimension = emb_np.shape[1]
        faiss.normalize_L2(emb_np)
        new_index = faiss.IndexFlatIP(embedding_dimension)
        new_index.add(emb_np)
        faiss_index = new_index
        print(f"✅ FAISS index built: {faiss_index.ntotal} vectors", flush=True)
    else:
        faiss_index = None
        embedding_dimension = None
        print("⚠️  FAISS index empty.", flush=True)

    # ── ساخت FAISS index انگلیسی — فقط آیتم‌هایی که question_en_norm دارند ──
    en_pairs = [
        (i, item["question_en_norm"])
        for i, item in enumerate(KNOWLEDGE_BASE)
        if item.get("question_en_norm") and item["question_en_norm"] in cached_embeddings
    ]

    if en_pairs:
        en_index_to_kb = [i for i, _ in en_pairs]
        emb_np_en = np.array([cached_embeddings[k] for _, k in en_pairs]).astype("float32")
        faiss.normalize_L2(emb_np_en)
        new_index_en = faiss.IndexFlatIP(emb_np_en.shape[1])
        new_index_en.add(emb_np_en)
        faiss_index_en = new_index_en
        print(f"✅ English FAISS index built: {faiss_index_en.ntotal} vectors", flush=True)
    else:
        faiss_index_en = None
        en_index_to_kb = []
        print("ℹ️  English FAISS index empty (no translated items yet).", flush=True)


# ═══════════════════════════════════════════════
#  Lifespan — startup / shutdown
# ═══════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global FALLBACK, prompt_config, _http, _rewarm_task
    global _job_queue, _slot_semaphore, _queue_worker_tasks

    # ── ساخت httpx client با connection pool ──
    _http = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        timeout=httpx.Timeout(60.0),
    )

    prompt_config = load_prompt_config()
    FALLBACK      = prompt_config.get("fallback", DEFAULT_FALLBACK)

    await rebuild_knowledge_base()

    # ── گرم کردن مدل chat در Ollama (لود در GPU قبل از اولین درخواست) ──
    print("🔥 Warming up Ollama chat model...", flush=True)
    warmup_result = await warmup_ollama_model()
    if warmup_result is not None:
        print("✅ Ollama chat model warmed up", flush=True)
    else:
        print("⚠️  Ollama chat model warmup failed (Ollama may be slow or unavailable)", flush=True)

    # ── شبکه ایمنی re-warm دوره‌ای — مستقل از OLLAMA_KEEP_ALIVE ──
    _rewarm_task = asyncio.create_task(periodic_ollama_rewarm())

    # ── صف /chat + worker pool — همان الگوی lifecycle که _rewarm_task استفاده می‌کند ──
    _job_queue = asyncio.Queue()
    _slot_semaphore = asyncio.Semaphore(PROCESSING_SLOTS)
    _queue_worker_tasks = [
        asyncio.create_task(chat_queue_worker(i)) for i in range(PROCESSING_SLOTS)
    ]
    print(f"✅ Chat queue workers started: {PROCESSING_SLOTS}", flush=True)

    yield

    # ── shutdown ──
    _rewarm_task.cancel()
    try:
        await _rewarm_task
    except asyncio.CancelledError:
        pass

    for t in _queue_worker_tasks:
        t.cancel()
    await asyncio.gather(*_queue_worker_tasks, return_exceptions=True)

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
    return {"status": "ok", "knowledge_base_items": len(KNOWLEDGE_BASE) if KNOWLEDGE_BASE is not None else 0}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    lang: str = "fa"


class FAQCreate(BaseModel):
    id: str | None = None
    category: str | None = None
    question: str
    answer: str
    question_en: str | None = None
    answer_en: str | None = None


class FAQUpdate(BaseModel):
    category: str | None = None
    question: str | None = None
    answer: str | None = None
    question_en: str | None = None
    answer_en: str | None = None


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
#  helper های lang-aware — برای دسترسی به فیلد صحیح (فارسی/انگلیسی) روی یک آیتم
#  اگر lang == "en" و آیتم ترجمه انگلیسی نداشته باشد، None برمی‌گردد
#  (یعنی آن آیتم در جستجوی انگلیسی نامرئی است — هرگز fallback به فارسی نمی‌شود)
# ═══════════════════════════════════════════════
def _lang_question_norm(item: dict, lang: str) -> str | None:
    if lang == "en":
        return item.get("question_en_norm") or None
    return item["question_norm"]


def _lang_search_text(item: dict, lang: str) -> str | None:
    if lang == "en":
        return item.get("search_text_en") or None
    return item["search_text"]


def _lang_question(item: dict, lang: str) -> str | None:
    if lang == "en":
        return item.get("question_en") or None
    return item["question"]


def _lang_answer(item: dict, lang: str) -> str | None:
    if lang == "en":
        return item.get("answer_en") or None
    return item["answer"]


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


def search_exact_knowledge(user_message: str, lang: str = "fa"):
    user_norm = normalize_text(user_message)
    best, best_score = None, 0.0
    for item in KNOWLEDGE_BASE:
        q_norm = _lang_question_norm(item, lang)
        if q_norm is None:
            continue  # این آیتم ترجمه انگلیسی ندارد — در جستجوی en نادیده گرفته می‌شود
        if user_norm == q_norm:
            return item, 1.0
        score = similarity(user_norm, q_norm)
        if score > best_score:
            best_score = score
            best = item
    if best and best_score >= 0.92:
        return best, best_score
    return None, best_score


async def search_hybrid_knowledge(user_message: str, top_k: int = 10, lang: str = "fa") -> list:
    """
    FAISS + keyword + fuzzy.
    embed_text_async یک‌بار await می‌شود — بقیه CPU-bound.
    ترتیب: اول embed (I/O)، بعد FAISS search (CPU).

    lang == "en": از faiss_index_en (فقط آیتم‌های ترجمه‌شده) و فیلدهای *_en استفاده می‌شود.
    lang == "fa" (پیش‌فرض): دقیقاً همان مسیر قبلی، بدون هیچ تغییر رفتاری.
    """
    active_index = faiss_index_en if lang == "en" else faiss_index
    index_to_kb  = en_index_to_kb if lang == "en" else None

    def _keyword_only_fallback():
        scored = []
        for item in KNOWLEDGE_BASE:
            search_text = _lang_search_text(item, lang)
            question    = _lang_question(item, lang)
            if search_text is None or question is None:
                continue
            score = (
                keyword_score(user_message, search_text) * 0.70
                + similarity(user_message, question) * 0.30
            )
            scored.append({"knowledge": item, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    if active_index is None or active_index.ntotal == 0:
        return _keyword_only_fallback()

    # ── async embed — این تنها I/O این تابع است ──
    # اگر Ollama در دسترس نباشد، به‌جای کرش کل request، به جستجوی keyword-only برمی‌گردیم
    try:
        embedding = await embed_text_async(user_message)
    except Exception as e:
        logger.warning(f"Ollama embedding unavailable (context=search_hybrid), falling back to keyword-only search: {type(e).__name__}: {e}")
        return _keyword_only_fallback()

    query_vec = np.array([embedding]).astype("float32")
    faiss.normalize_L2(query_vec)
    k = min(top_k, active_index.ntotal)
    D, I = active_index.search(query_vec, k)

    scored = []
    for idx, emb_score in zip(I[0], D[0]):
        if idx == -1:
            continue
        kb_idx = index_to_kb[idx] if index_to_kb is not None else idx
        if kb_idx >= len(KNOWLEDGE_BASE):
            continue
        item = KNOWLEDGE_BASE[kb_idx]
        search_text = _lang_search_text(item, lang)
        question    = _lang_question(item, lang)
        if search_text is None or question is None:
            continue
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
            booth_words = ["booth", "which hall", "booth number", "which stand", "stand number"]
            info_words  = ["address", "phone", "contact", "where", "email"]
        else:
            booth_words = ["غرفه", "کدوم سالن", "شماره غرفه", "کدومه"]
            info_words  = ["آدرس", "تلفن", "شماره تماس", "کجاست"]

        wants_booth_only = (
            any(w in user_norm for w in booth_words)
            and not any(w in user_norm for w in info_words)
        )

        if lang == "en":
            parts = [f"🏢 Company: {data['company']}"]
            if data.get("booth"):
                parts.append(f"📍 Booth: {data['booth']}")
            if wants_booth_only:
                return "\n".join(parts)
            if data.get("extra"):
                return "\n".join(parts) + "\n\n📞 Additional info:\n" + data["extra"]
            return "\n".join(parts)

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

    options = "".join(
        f"{i}. {_lang_question(c['knowledge'], lang) or c['knowledge']['question']}\n"
        for i, c in enumerate(candidates, 1)
    )

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
        context="select_candidate",
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
        logger.warning(f"Ollama returned unparseable candidate selection {raw!r} — falling back to top FAISS candidate for query {user_message[:80]!r}")
        return candidates[0] if candidates else None

    # raw is None — call_ollama_async already logged the timeout/error detail above
    logger.warning(f"Ollama candidate-selection unavailable — falling back to top FAISS candidate for query {user_message[:80]!r}")
    return candidates[0] if candidates else None


# ═══════════════════════════════════════════════
#  پایپ‌لاین اصلی چت — بدون تغییر منطقی، فقط جابه‌جا شده از داخل endpoint
#  به یک تابع مستقل تا هم مسیر inline و هم queue worker از آن استفاده کنند.
# ═══════════════════════════════════════════════
async def run_chat_pipeline(req: ChatRequest) -> dict:
    user_message = req.message.strip()
    lang = "en" if req.lang == "en" else "fa"
    if not user_message:
        return {"answer": FALLBACK, "source": "empty"}

    # ── بررسی سوالات meta درباره تاریخچه ──
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
        await log_chat_interaction(user_message, ans, "meta", 1.0)
        return {"answer": ans, "source": "meta"}

    # ۱. Query Enricher (sync CPU — روی thread جدا تا event loop تک‌پردازه را در بار همزمان بلاک نکند)
    search_query = await asyncio.to_thread(contextualize_question, user_message, req.history)

    # ۲. Example search (sync CPU — روی thread جدا)
    example_answer, _ = await asyncio.to_thread(search_examples, search_query)
    if example_answer:
        await log_chat_interaction(user_message, example_answer, "example", 1.0, search_query)
        return {"answer": example_answer, "source": "example"}

    # ۳. Exact / fuzzy search (sync CPU — روی thread جدا؛ O(n) روی کل KNOWLEDGE_BASE)
    exact_item, exact_score = await asyncio.to_thread(search_exact_knowledge, search_query, lang=lang)
    if exact_item:
        ans = _lang_answer(exact_item, lang)
        if exact_item.get("is_directory"):
            ans = format_directory_response(user_message, ans, lang=lang)
        await log_chat_interaction(user_message, ans, "exact", exact_score, _lang_question(exact_item, lang))
        return {"answer": ans, "source": "exact"}

    # ۴. FAISS + hybrid (async — شامل embed I/O)
    candidates = await search_hybrid_knowledge(search_query, top_k=10, lang=lang)
    if not candidates:
        await log_chat_interaction(user_message, FALLBACK, "fallback", 0.0)
        return {"answer": FALLBACK, "source": "fallback"}

    print(f"📊 scores: {[round(c['score'],3) for c in candidates]}", flush=True)

    # ۵. Threshold filter
    MIN_SCORE  = 0.50
    candidates = [c for c in candidates if c["score"] >= MIN_SCORE]
    if not candidates:
        await log_chat_interaction(user_message, FALLBACK, "fallback_threshold", 0.0)
        return {"answer": FALLBACK, "source": "fallback"}

    # ۶. Ollama انتخاب (async — I/O)
    selected = await select_best_candidate(search_query, candidates[:5], lang=lang)
    if not selected:
        await log_chat_interaction(user_message, FALLBACK, "fallback", 0.0)
        return {"answer": FALLBACK, "source": "fallback"}

    # ۷. فرمت و برگشت
    ans = _lang_answer(selected["knowledge"], lang)
    if selected["knowledge"].get("is_directory"):
        ans = format_directory_response(user_message, ans, lang=lang)

    await log_chat_interaction(user_message, ans, "rag", selected["score"], _lang_question(selected["knowledge"], lang))
    return {"answer": ans, "source": "rag"}


# ═══════════════════════════════════════════════
#  صف /chat — helperها
# ═══════════════════════════════════════════════
def _estimate_wait_seconds(position: int) -> int:
    """تخمین تقریبی — position بر اساس PROCESSING_SLOTS دسته‌بندی و در AVG_PIPELINE_SECONDS ضرب می‌شود."""
    batches_ahead = -(-position // PROCESSING_SLOTS)  # ceil division
    return batches_ahead * AVG_PIPELINE_SECONDS


def _cleanup_stale_queue_results():
    """پاک‌سازی نتایج قدیمی از _queue_results — جلوگیری از رشد نامحدود حافظه."""
    now = time.monotonic()
    stale_ids = [
        qid for qid, entry in _queue_results.items()
        if (entry["state"] == "done" and now - entry["completed_at"] > QUEUE_RESULT_TTL_SECONDS)
        # شبکه ایمنی: یک job که هرگز worker آن را برنداشته (نباید عملاً رخ دهد چون
        # worker pool ثابت و FIFO است) هم دیر یا زود پاک می‌شود.
        or (entry["state"] == "queued" and now - entry["enqueued_at"] > MAX_WAIT_SECONDS + QUEUE_RESULT_TTL_SECONDS)
    ]
    for qid in stale_ids:
        del _queue_results[qid]


async def chat_queue_worker(worker_id: int):
    """
    یکی از PROCESSING_SLOTS workerهای ثابت — صف را FIFO تخلیه می‌کند.
    همان pipeline دقیقاً مثل مسیر inline اجرا می‌شود (run_chat_pipeline بدون تغییر).
    قبل از پردازش، منتظر آزاد شدن یک slot از _slot_semaphore می‌ماند — یعنی مجموع
    اجرای همزمان pipeline (inline + queue) هرگز از PROCESSING_SLOTS بیشتر نمی‌شود.
    """
    global _dequeued_count
    while True:
        job = await _job_queue.get()
        _dequeued_count += 1
        try:
            await _slot_semaphore.acquire()
            try:
                result = await run_chat_pipeline(job.req)
                _queue_results[job.queue_id] = {
                    "state": "done",
                    "answer": result.get("answer"),
                    "source": result.get("source"),
                    "completed_at": time.monotonic(),
                }
            finally:
                _slot_semaphore.release()
        except Exception as e:
            logger.warning(f"Queue worker {worker_id} pipeline error (queue_id={job.queue_id}): {type(e).__name__}: {e}")
            _queue_results[job.queue_id] = {
                "state": "done",
                "answer": FALLBACK,
                "source": "error",
                "completed_at": time.monotonic(),
            }
        finally:
            _job_queue.task_done()


# ═══════════════════════════════════════════════
#  endpoint اصلی — async
#  slot آزاد → پردازش inline (دقیقاً مثل قبل، بدون هیچ latency اضافه).
#  بدون slot → یا صف (تا سقف MAX_QUEUE_DEPTH) یا fast-fail "busy".
# ═══════════════════════════════════════════════
@app.post("/chat")
async def chat(req: ChatRequest):
    global _next_seq

    if not _slot_semaphore.locked():
        # هیچ await بین چک بالا و acquire زیر نیست — پس این acquire تضمینی و فوری است
        # (asyncio تک‌رشته‌ای/cooperative است، پس هیچ coroutine دیگری نمی‌تواند بین این دو خط اجرا شود)
        await _slot_semaphore.acquire()
        try:
            return await run_chat_pipeline(req)
        finally:
            _slot_semaphore.release()

    _cleanup_stale_queue_results()

    if _job_queue.qsize() >= MAX_QUEUE_DEPTH:
        return {"answer": BUSY_MESSAGE_FA, "source": "busy"}

    _next_seq += 1
    seq = _next_seq
    queue_id = secrets.token_hex(8)
    now = time.monotonic()
    position = seq - _dequeued_count

    _queue_results[queue_id] = {"state": "queued", "seq": seq, "enqueued_at": now}
    await _job_queue.put(_QueueJob(queue_id=queue_id, req=req, seq=seq, enqueued_at=now))

    return {
        "status": "queued",
        "queue_id": queue_id,
        "position": position,
        "estimated_wait_seconds": _estimate_wait_seconds(position),
    }


@app.get("/chat/status/{queue_id}")
async def chat_status(queue_id: str):
    _cleanup_stale_queue_results()

    entry = _queue_results.get(queue_id)
    if entry is None:
        # queue_id نامعتبر/منقضی‌شده — یعنی یا هرگز وجود نداشته یا مدت‌ها پیش پاک‌سازی شده
        return {"status": "busy"}

    if entry["state"] == "done":
        return {"status": "done", "answer": entry["answer"], "source": entry["source"]}

    elapsed = time.monotonic() - entry["enqueued_at"]
    if elapsed >= MAX_WAIT_SECONDS:
        return {"status": "busy"}

    position = max(entry["seq"] - _dequeued_count, 1)
    return {"status": "queued", "position": position}


# ═══════════════════════════════════════════════
#  endpoint لاگ — برای پنل مدیریت
# ═══════════════════════════════════════════════
@app.get("/logs")
async def get_logs(source: str = None, limit: int = 500):
    logs = []
    if not LOG_FILE_PATH.exists():
        return {"logs": []}
    try:
        with open(LOG_FILE_PATH, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if source and row.get("Source", "").lower() != source.lower():
                    continue
                logs.append({
                    "timestamp":       row.get("Timestamp", ""),
                    "user_message":    row.get("User_Message", ""),
                    "bot_answer":      row.get("Bot_Answer", ""),
                    "source":          row.get("Source", ""),
                    "score":           row.get("Score", "0"),
                    "matched_question": row.get("Matched_Question", ""),
                })
    except Exception as e:
        return {"logs": [], "error": str(e)}
    return {"logs": logs[-limit:]}


# ═══════════════════════════════════════════════
#  Admin CRUD — مدیریت FAQ (Postgres)
#  همه‌ی endpointها با هدر X-Admin-Key محافظت می‌شوند.
#  بعد از هر تغییر، embeddings/FAISS بلافاصله و خودکار rebuild می‌شود.
# ═══════════════════════════════════════════════
def _faq_row_to_dict(row) -> dict:
    return {
        "id": row[0], "category": row[1], "question": row[2], "answer": row[3],
        "question_en": row[4], "answer_en": row[5],
    }


def _generate_faq_id(existing_ids: set) -> str:
    while True:
        candidate = "".join(secrets.choice(FAQ_ID_ALPHABET) for _ in range(8))
        if candidate not in existing_ids:
            return candidate


@app.get("/admin/faq")
async def admin_list_faq(_: None = Depends(verify_admin_key)):
    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, category, question, answer, question_en, answer_en "
                "FROM faq ORDER BY created_at"
            )
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    return [_faq_row_to_dict(r) for r in rows]


@app.post("/admin/faq")
async def admin_create_faq(payload: FAQCreate, _: None = Depends(verify_admin_key)):
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
                INSERT INTO faq (id, category, question, answer, question_en, answer_en)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id, category, question, answer, question_en, answer_en
                """,
                (new_id, payload.category, payload.question, payload.answer,
                 payload.question_en, payload.answer_en),
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

    await rebuild_knowledge_base()
    return _faq_row_to_dict(row)


@app.put("/admin/faq/{faq_id}")
async def admin_update_faq(faq_id: str, payload: FAQUpdate, _: None = Depends(verify_admin_key)):
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    set_clauses = [f"{field} = %s" for field in updates] + ["updated_at = NOW()"]
    values = list(updates.values()) + [faq_id]

    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE faq SET {', '.join(set_clauses)} WHERE id = %s "
                f"RETURNING id, category, question, answer, question_en, answer_en",
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

    await rebuild_knowledge_base()
    return _faq_row_to_dict(row)


@app.delete("/admin/faq/{faq_id}")
async def admin_delete_faq(faq_id: str, _: None = Depends(verify_admin_key)):
    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM faq WHERE id = %s", (faq_id,))
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

    await rebuild_knowledge_base()
    return {"deleted": True}


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

_COMPANY_ALL_COLUMNS = COMPANY_FIELDS + ["event_id"]
_COMPANY_INSERT_SQL = (
    f"INSERT INTO companies ({', '.join(_COMPANY_ALL_COLUMNS)}) "
    f"VALUES ({', '.join(['%s'] * len(_COMPANY_ALL_COLUMNS))}) "
    f"ON CONFLICT (id) DO UPDATE SET "
    + ", ".join(f"{col} = EXCLUDED.{col}" for col in COMPANY_FIELDS if col != "id")
    + ", event_id = EXCLUDED.event_id, synced_at = NOW()"
)


@app.get("/admin/companies")
async def admin_list_companies(_: None = Depends(verify_admin_key)):
    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM companies ORDER BY brand_name_fa")
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        if conn:
            conn.close()

    return [dict(r) for r in rows]


@app.post("/admin/companies/sync")
async def admin_sync_companies(payload: dict = Body(...), _: None = Depends(verify_admin_key)):
    event_id = payload.get("event_id")
    companies = payload.get("companies")

    if not isinstance(event_id, int) or isinstance(event_id, bool):
        raise HTTPException(status_code=400, detail="'event_id' must be an integer")
    if not isinstance(companies, list):
        raise HTTPException(status_code=400, detail="'companies' must be a list")
    for i, c in enumerate(companies):
        if not isinstance(c, dict) or "id" not in c:
            raise HTTPException(status_code=400, detail=f"companies[{i}] must be an object with an 'id' field")

    conn = None
    try:
        conn = get_faq_db_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM companies WHERE event_id != %s", (event_id,))
            for c in companies:
                values = []
                for field in COMPANY_FIELDS:
                    value = c.get(field)
                    if field in COMPANY_JSON_FIELDS and value is not None:
                        value = Json(value)
                    elif field in COMPANY_BOOLEAN_FIELDS and value is None:
                        value = False
                    values.append(value)
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

    await rebuild_knowledge_base()
    return {"synced": len(companies), "event_id": event_id}
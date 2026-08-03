FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# نصب پکیج‌ها با محدود کردن نامپای به نسخه 1.x جهت سازگاری با فایس قدیمی
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
    -i https://mirror.abrha.net/repository/pypi/simple \
    --trusted-host mirror.abrha.net \
    fastapi uvicorn pydantic httpx "numpy<2.0.0" "faiss-cpu==1.7.4"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

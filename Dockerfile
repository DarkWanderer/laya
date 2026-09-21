FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=8 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Install CPU-only PyTorch before Laya so pip can reuse it.
RUN pip install --no-cache-dir 'torch==2.9.1+cpu' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir 'laya==0.3.5' 'fastapi>=0.115,<1' 'uvicorn>=0.34,<1' 'pytest>=8,<10' 'httpx>=0.27,<1'

COPY download_model.py /app/download_model.py
RUN HF_HUB_OFFLINE=0 python /app/download_model.py \
    && rm -rf /opt/model/.cache /root/.cache

COPY app /app/app
COPY tests /app/tests
COPY bench /app/bench

RUN useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin laya
USER 10001:10001
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]

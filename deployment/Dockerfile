FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 10001 quantai
WORKDIR /app
COPY requirements-core.txt .
RUN pip install --upgrade pip && pip install -r requirements-core.txt
COPY . .
RUN chown -R quantai:quantai /app
USER quantai
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000","--workers","1"]

FROM runtime AS api


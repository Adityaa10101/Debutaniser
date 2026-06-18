# ─────────────────────────────────────────────────────────────────
#  Debutanizer Soft Sensor — Dockerfile
#  Base: slim Python 3.11 (lightweight, well-supported)
#  Stack: XGBoost | Streamlit | Plotly | Pandas | scikit-learn
# ─────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# OS-level build dependencies (needed by numpy / pandas C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Layer-cache optimisation ──────────────────────────────────────
# Copy requirements first — Docker only re-installs packages if
# requirements.txt changes, not every time app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────
# .dockerignore keeps raw CSVs and caches out of the image
COPY . .

# ── Streamlit configuration ───────────────────────────────────────
EXPOSE 8501

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# ── Healthcheck — Docker will mark container unhealthy if app dies ─
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# ── Launch ────────────────────────────────────────────────────────
CMD ["python", "-m", "streamlit", "run", "app.py"]

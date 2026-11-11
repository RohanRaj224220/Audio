# Hosted deployment image (Render, Railway, Hugging Face Spaces, any Docker host).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8501 \
    MAX_UPLOAD_MB=200

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app . .
RUN mkdir -p workspace/jobs logs && chown -R app:app /app
USER app

# Bake the vocal-separation model into the image so it is not re-downloaded on
# every restart. Build with --build-arg PRELOAD_MODEL=0 to skip.
ARG PRELOAD_MODEL=1
RUN if [ "$PRELOAD_MODEL" = "1" ]; then python -c "import audio_separator as s; s.load_model(); s.ensure_best_models()"; fi

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8501\")}/_stcore/health')"
CMD streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT}

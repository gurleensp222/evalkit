FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin evalkit

# Dependencies only, so edits to the package do not rebuild this layer.
COPY pyproject.toml ./
RUN python -c "import tomllib,subprocess; from pathlib import Path; deps=tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))['project']['dependencies']; subprocess.check_call(['pip','install','--no-cache-dir',*deps])"

COPY README.md ./
COPY evalkit ./evalkit
COPY datasets ./datasets
RUN pip install --no-cache-dir --no-deps . \
    && mkdir -p /app/runs \
    && chown -R evalkit:evalkit /app

USER evalkit

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "evalkit.api:app", "--host", "0.0.0.0", "--port", "8000"]

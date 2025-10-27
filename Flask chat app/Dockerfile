FROM python:3.11-slim

ARG PIP_EXTRA_INDEX_URL=""
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./

# Install system dependencies required for common ML libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    pip install --no-cache-dir --upgrade pip && \
    if [ -n "$PIP_EXTRA_INDEX_URL" ]; then \
        pip install --no-cache-dir -r requirements.txt --extra-index-url "$PIP_EXTRA_INDEX_URL"; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi && \
    pip install --no-cache-dir gunicorn && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY . .

ENV FLASK_APP_PATH=/app
ENV FLASK_TEMPLATE_PATH=/app/templates
ENV FLASK_STATIC_PATH=/app/static

EXPOSE 5000

CMD ["gunicorn", "--workers", "2", "--threads", "4", "--bind", "0.0.0.0:5000", "app:app"]

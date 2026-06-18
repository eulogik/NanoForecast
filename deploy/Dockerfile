FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nanoforecast/ nanoforecast/
COPY pyproject.toml .
RUN pip install -e .

COPY checkpoints/ checkpoints/

EXPOSE 8000
CMD ["uvicorn", "deploy.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]

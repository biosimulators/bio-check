FROM python:3.11-slim

WORKDIR /src

COPY ../../dockerfile-assets/requirements-worker.txt .

RUN pip install --no-cache-dir -r requirements.api.txt

COPY bio_check/worker .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

 # Api container


# FROM python:3.11-slim-buster
FROM docker.io/spacebearamadeus/verification-service-base:latest

EXPOSE 3001

RUN pip install --no-cache-dir -r ./dockerfile-assets/requirements-api.txt

CMD ["uvicorn", "verification_service.api.main:app", "--host", "0.0.0.0", "--port", "3001", "--reload"]


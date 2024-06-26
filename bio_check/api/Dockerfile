 # Api container

FROM ghcr.io/biosimulators/bio-check-base:latest

EXPOSE 3001

# COPY . .

RUN pip install --no-cache-dir -r ./requirements.api.txt

CMD ["uvicorn", "bio_check.api.main:app", "--host", "0.0.0.0", "--port", "3001", "--reload"]


 # Api container

FROM ghcr.io/biosimulators/bio-check-base:latest

# RUN useradd -r -g appgroup api

# USER api

EXPOSE 3001

RUN poetry add tensorstore aiofiles aiobotocore pyyaml httpx uvicorn

COPY . /app/bio_check

CMD ["uvicorn", "bio_check.api.main:app", "--host", "0.0.0.0", "--port", "3001", "--reload"]

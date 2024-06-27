 # Api container

FROM ghcr.io/biosimulators/bio-check-base:latest

# RUN useradd -r -g appgroup api

# USER api

RUN mkdir /app/bio_check/api

COPY . /app/bio_check/api

EXPOSE 3001

RUN poetry add tensorstore aiofiles aiobotocore pyyaml httpx uvicorn

CMD ["poetry", "run", "uvicorn", "bio_check.api.main:app", "--host", "0.0.0.0", "--port", "3001", "--reload"]

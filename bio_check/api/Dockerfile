 # Api container

FROM ghcr.io/biosimulators/bio-check-base:latest

ENV LIB=api
ENV API_ROOT=/app/bio_check/$LIB
RUN mkdir $API_ROOT
COPY . $API_ROOT

COPY data_model.py /app/bio_check/api/data_model.py

EXPOSE 3001

RUN poetry add tensorstore aiofiles aiobotocore pyyaml httpx uvicorn

CMD ["poetry", "run", "uvicorn", "bio_check.api.main:app", "--host", "0.0.0.0", "--port", "3001", "--reload"]

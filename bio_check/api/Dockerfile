 # Api container

FROM ghcr.io/biosimulators/bio-check-base:latest

ENV LIB=api
ENV API_ROOT=/app/bio_check/$LIB

RUN mkdir $API_ROOT
RUN mkdir /app/bio_check/data_model

COPY data_model.py /app/bio_check/data_model/api.py
COPY ../shared.py /app/bio_check/data_model/shared.py
COPY . $API_ROOT

EXPOSE 3001

RUN poetry add tensorstore aiofiles aiobotocore pyyaml httpx uvicorn

CMD ["poetry", "run", "uvicorn", "bio_check.api.main:app", "--host", "0.0.0.0", "--port", "3001", "--reload"]

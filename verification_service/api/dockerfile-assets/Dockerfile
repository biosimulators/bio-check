 # Api container


FROM python:3.11-slim-buster

WORKDIR /src

COPY .. /src
COPY requirements.txt /src

EXPOSE 3001

RUN "pip install -r ./requirements.txt"

CMD ["python3", "uvicorn", "src.main:app", "--reload"]


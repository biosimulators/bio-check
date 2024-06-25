 # Api container


FROM python:3.11-slim-buster

WORKDIR /src

COPY requirements.txt .

RUN cd ..

COPY . .

EXPOSE 3001

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r ./requirements.txt

RUN echo "This is src: "
RUN ls -al

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3001", "--reload"]


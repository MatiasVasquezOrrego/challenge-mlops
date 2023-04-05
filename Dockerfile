FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY . /src /app/
COPY . /models /app/
COPY api.py /app/

EXPOSE 8000

ENTRYPOINT uvicorn api:app

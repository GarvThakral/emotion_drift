FROM python:3.10-slim
WORKDIR /api
COPY apiPy.py requirements.txt ./
RUN pip install -r ./api/requirements.txt

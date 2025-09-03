FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --progress-bar=on -v -r requirements.txt

# Copy code
COPY  app.py .

EXPOSE 7860

# Run FastAPI

ENV PORT=7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

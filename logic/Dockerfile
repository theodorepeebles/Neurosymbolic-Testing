FROM python:3.12-slim

# Print logs immediately so they show up in Cloud Run.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY pipeline/ pipeline/

WORKDIR /app/pipeline
CMD ["python", "chatbox.py"]

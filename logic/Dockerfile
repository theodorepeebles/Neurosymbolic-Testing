FROM python:3.12-slim

# Print logs immediately so they show up in Cloud Run.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Same layout as logic/, so the server's ../pipeline and ../data paths work.
COPY pipeline/ pipeline/
COPY data/demo_model_outputs.json data/sft_test.jsonl data/
COPY data/plots/*.png data/plots/
COPY replay/ replay/

WORKDIR /app/replay
CMD ["python", "replay_server.py"]

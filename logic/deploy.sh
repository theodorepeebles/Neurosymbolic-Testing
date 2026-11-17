#!/usr/bin/env bash
# Build logic/ with Cloud Build and deploy the replay demo to Cloud Run.
set -euo pipefail

# Run from the repo root so --source logic resolves, wherever this script is called from.
cd "$(dirname "$0")/.."

gcloud run deploy ns-logic-demo \
  --source logic \
  --region us-central1 \
  --allow-unauthenticated \
  --cpu 1 \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 60 \
  --set-env-vars HOST=0.0.0.0

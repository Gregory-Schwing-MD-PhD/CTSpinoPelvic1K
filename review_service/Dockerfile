# HuggingFace Space (Docker SDK) for the CTSpinoPelvic1K review service.
# Build context = repo root:  docker build -f review_service/Dockerfile .
FROM python:3.11-slim

WORKDIR /app
COPY review_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# The service code + the Phase-1 review package it depends on.
COPY review_service/ /app/review_service/
COPY scripts/review/ /app/scripts/review/

ENV PYTHONUNBUFFERED=1
# HF Spaces route to port 7860.
EXPOSE 7860
CMD ["uvicorn", "review_service.app:app", "--host", "0.0.0.0", "--port", "7860"]

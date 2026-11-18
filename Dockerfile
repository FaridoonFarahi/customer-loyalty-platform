# Scoring service for the coupon redemption model.
#   docker build -t loyalty-scoring .
#   docker run -p 8000:8000 loyalty-scoring      # then open http://127.0.0.1:8000/docs
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# The service needs the code and the saved model. It does not need the raw data.
COPY src/ src/
COPY models/redemption_model.joblib models/redemption_model_metadata.json models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

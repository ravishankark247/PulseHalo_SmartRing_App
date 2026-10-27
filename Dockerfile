FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY server.py ml_engine.py index.html ./

EXPOSE 8080

CMD ["python", "server.py"]

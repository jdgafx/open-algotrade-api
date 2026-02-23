FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY start.py .
COPY .env.example ./.env
EXPOSE 8000

CMD ["python", "start.py"]

FROM python:3.11-slim

WORKDIR /app

# System deps for C extensions (hmmlearn, numpy wheels, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build tools after pip install to shrink image
RUN apt-get purge -y gcc g++ && apt-get autoremove -y

COPY src/ ./src/
COPY start.py .
COPY .env.example ./.env

EXPOSE 8000

CMD ["python", "start.py"]

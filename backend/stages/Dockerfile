FROM python:3.11-slim

WORKDIR /app

# 1. Prevent Python from buffering stdout/stderr (CRITICAL for Streaming)
ENV PYTHONUNBUFFERED=1

# Install system dependencies including PostgreSQL client libraries
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use python main.py to respect the __main__ block
CMD ["python", "main.py"]
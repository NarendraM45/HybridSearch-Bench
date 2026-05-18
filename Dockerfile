# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies to the user site-packages
RUN pip install --no-cache-dir --user -r requirements.txt
RUN python -m spacy download en_core_web_sm


# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Ensure user site-packages are on the PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy installed dependencies from the builder
COPY --from=builder /root/.local /root/.local

# Copy source code
COPY . /app

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)"

ENTRYPOINT ["python", "-m", "cli"]
CMD ["serve"]

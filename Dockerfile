# 1. Build dependencies
FROM python:3.11-slim AS builder
WORKDIR /app

# Install uv
RUN pip install uv

COPY requirements.txt .
# Install dependencies
RUN uv pip install --no-cache --system -r requirements.txt

# 2. Final image
FROM python:3.11-slim
WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

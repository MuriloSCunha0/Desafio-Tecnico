# ============================================================
# Dockerfile — Banco Ágil Multi-Agent Banking System
# ============================================================

FROM python:3.11-slim

# Evitar prompts interativos
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential tzdata && \
    rm -rf /var/lib/apt/lists/*

# Instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar diretórios de dados do backend
RUN mkdir -p /app/backend/data /app/backend/db


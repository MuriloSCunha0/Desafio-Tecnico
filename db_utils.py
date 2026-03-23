"""
db_utils.py — Utilitários de leitura e escrita para os arquivos CSV do Banco Ágil.

Funções auxiliares para manipular os dados de clientes, limites,
score e solicitações armazenados em arquivos CSV locais.
"""

import csv
import os
from datetime import datetime
from typing import Optional

from logger import get_logger

logger = get_logger("db_utils")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CLIENTS_CSV = os.path.join(DATA_DIR, "clients.csv")
SCORE_LIMITE_CSV = os.path.join(DATA_DIR, "score_limite.csv")
SOLICITACOES_CSV = os.path.join(DATA_DIR, "solicitacoes_aumento_limite.csv")


# ============================================================
# Leitura / Escrita genérica
# ============================================================

def read_csv(path: str) -> list[dict]:
    """Lê um arquivo CSV e retorna uma lista de dicionários."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        logger.error("read_csv | path=%s | %s", path, e, exc_info=True)
        return []


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Escreve uma lista de dicionários em um arquivo CSV."""
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        logger.error("write_csv | path=%s | %s", path, e, exc_info=True)
        raise


def append_csv(path: str, row: dict, fieldnames: list[str]) -> None:
    """Adiciona uma linha ao final de um arquivo CSV existente."""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    try:
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        logger.error("append_csv | path=%s | row=%s | %s", path, row, e, exc_info=True)
        raise


# ============================================================
# Clientes (clients.csv) — agora contém score e current_limit
# ============================================================

CLIENT_FIELDNAMES = ["cpf", "name", "date_of_birth", "score", "current_limit"]


def find_client(cpf: str) -> Optional[dict]:
    """Busca um cliente pelo CPF no arquivo clients.csv."""
    cpf_clean = cpf.replace(".", "").replace("-", "").strip()
    clients = read_csv(CLIENTS_CSV)
    for client in clients:
        if client["cpf"].strip() == cpf_clean:
            return client
    return None


def update_client(cpf: str, new_data: dict) -> bool:
    """
    Atualiza os dados de um cliente no clients.csv.

    Args:
        cpf: CPF do cliente (pode conter pontuação).
        new_data: Dicionário com campos a atualizar
                  (ex: {"score": "750", "current_limit": "10000.00"}).

    Returns:
        True se o registro foi encontrado e atualizado, False caso contrário.
    """
    cpf_clean = cpf.replace(".", "").replace("-", "").strip()
    clients = read_csv(CLIENTS_CSV)
    found = False

    for row in clients:
        if row["cpf"].strip() == cpf_clean:
            row.update(new_data)
            found = True
            break

    if found:
        write_csv(CLIENTS_CSV, clients, CLIENT_FIELDNAMES)

    return found


# ============================================================
# Tabela Score ↔ Limite (score_limite.csv)
# ============================================================

def get_max_limit_for_score(score: int) -> float:
    """
    Consulta a tabela score_limite.csv e retorna o limite máximo
    permitido para o score informado.

    A tabela é percorrida do maior para o menor score_minimo.
    O primeiro registro cujo score_minimo ≤ score fornecido é usado.
    """
    rows = read_csv(SCORE_LIMITE_CSV)
    # Ordenar do maior para o menor score_minimo
    rows.sort(key=lambda r: int(r["score_minimo"]), reverse=True)

    for row in rows:
        if score >= int(row["score_minimo"]):
            return float(row["limite_maximo"])

    # Fallback: menor faixa
    if rows:
        return float(rows[-1]["limite_maximo"])
    return 0.0


# ============================================================
# Solicitações de Aumento (solicitacoes_aumento_limite.csv)
# ============================================================

SOLICITACAO_FIELDNAMES = [
    "cpf_cliente",
    "data_hora_solicitacao",
    "limite_atual",
    "novo_limite_solicitado",
    "status_pedido",
]


def log_solicitacao(cpf: str, limite_atual: float,
                    novo_limite: float, status: str) -> None:
    """
    Registra uma solicitação de aumento de limite no CSV de solicitações.

    Args:
        cpf: CPF do cliente.
        limite_atual: Limite antes da solicitação.
        novo_limite: Valor solicitado pelo cliente.
        status: 'pendente', 'aprovado' ou 'rejeitado'.
    """
    row = {
        "cpf_cliente": cpf.replace(".", "").replace("-", "").strip(),
        "data_hora_solicitacao": datetime.now().isoformat(),
        "limite_atual": f"{limite_atual:.2f}",
        "novo_limite_solicitado": f"{novo_limite:.2f}",
        "status_pedido": status,
    }
    append_csv(SOLICITACOES_CSV, row, SOLICITACAO_FIELDNAMES)

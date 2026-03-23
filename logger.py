"""
logger.py — Configuração centralizada de log para o Banco Ágil.

Grava erros em logs/errors.log com rotação automática (5 MB, 3 backups).
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOGS_DIR, "errors.log")

os.makedirs(LOGS_DIR, exist_ok=True)

_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=3,
    encoding="utf-8",
)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
))


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger configurado para gravar em logs/errors.log."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.ERROR)
        logger.addHandler(_handler)
    return logger

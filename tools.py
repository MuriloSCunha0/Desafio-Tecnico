"""
tools.py — Tools (ferramentas) para os agentes do Banco Ágil.

Cada tool é decorada com @tool do LangChain e encapsula uma operação
específica do domínio bancário: autenticação, crédito, entrevista, câmbio.
"""

import requests
from langchain_core.tools import tool

from logger import get_logger
from db_utils import (
    find_client,
    update_client,
    get_max_limit_for_score,
    log_solicitacao,
)

logger = get_logger("tools")


# ============================================================
# Tool: Autenticação
# ============================================================

@tool
def authenticate_user(cpf: str, date_of_birth: str) -> str:
    """
    Autentica um usuário verificando CPF e data de nascimento.

    Args:
        cpf: CPF do cliente (apenas números, ex: 12345678901).
        date_of_birth: Data de nascimento no formato DD/MM/AAAA (ex: 15/05/1990) ou AAAA-MM-DD (ex: 1990-05-15).

    Returns:
        Mensagem indicando sucesso ou falha na autenticação, incluindo o nome do cliente.
    """
    try:
        client = find_client(cpf)

        if client is None:
            return "FALHA: CPF não encontrado no sistema. Verifique o número informado."

        # Converter DD/MM/AAAA ou DD-MM-AAAA para AAAA-MM-DD se necessário
        sep = "/" if "/" in date_of_birth else ("-" if "-" in date_of_birth else None)
        if sep:
            parts = date_of_birth.split(sep)
            # Só reordena se o formato for DD-MM-AAAA (ano com 4 dígitos no índice 2)
            if len(parts) == 3 and len(parts[2]) == 4:
                date_of_birth = f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"

        if client["date_of_birth"].strip() == date_of_birth.strip():
            return f"SUCESSO: Usuário autenticado! Bem-vindo(a), {client['name']}!"
        else:
            return "FALHA: Data de nascimento não confere com o CPF informado."
    except Exception as e:
        logger.error("authenticate_user | cpf=%s | %s", cpf, e, exc_info=True)
        return f"ERRO: Falha ao acessar a base de clientes: {str(e)}"


# ============================================================
# Tool: Consulta de Limite
# ============================================================

@tool
def check_limit(cpf: str) -> str:
    """
    Consulta o limite de crédito atual de um cliente.

    Args:
        cpf: CPF do cliente (apenas números).

    Returns:
        Informações sobre o limite atual e score do cliente.
    """
    try:
        client = find_client(cpf)

        if client is None:
            return "ERRO: Não foi possível encontrar dados de crédito para este CPF."

        current_limit = float(client["current_limit"])
        score = int(client["score"])

        return (
            f"Dados de crédito encontrados:\n"
            f"- Limite atual: R$ {current_limit:,.2f}\n"
            f"- Score de crédito: {score} pontos"
        )
    except Exception as e:
        logger.error("check_limit | cpf=%s | %s", cpf, e, exc_info=True)
        return f"ERRO: Falha ao consultar dados de crédito: {str(e)}"


# ============================================================
# Tool: Solicitação de Aumento de Limite
# ============================================================

@tool
def request_limit_increase(cpf: str, requested_value: float) -> str:
    """
    Solicita um aumento de limite de crédito para o cliente.
    Registra a solicitação em solicitacoes_aumento_limite.csv e
    verifica a tabela score_limite.csv para aprovação.

    Args:
        cpf: CPF do cliente (apenas números).
        requested_value: Valor de limite desejado em reais (ex: 10000.00).

    Returns:
        Resultado da solicitação (aprovado ou rejeitado com orientações).
    """
    try:
        client = find_client(cpf)

        if client is None:
            return "ERRO: Não foi possível encontrar dados de crédito para este CPF."

        current_limit = float(client["current_limit"])
        score = int(client["score"])

        # Consultar tabela score_limite.csv para obter o limite máximo permitido
        max_allowed = get_max_limit_for_score(score)

        if requested_value <= max_allowed:
            # APROVADO
            status = "aprovado"
            log_solicitacao(cpf, current_limit, requested_value, status)
            update_client(cpf, {"current_limit": f"{requested_value:.2f}"})
            return (
                f"APROVADO: Seu limite foi atualizado para R$ {requested_value:,.2f}.\n"
                f"Score atual: {score} pontos.\n"
                f"Limite máximo permitido pelo seu score: R$ {max_allowed:,.2f}."
            )
        else:
            # REJEITADO
            status = "rejeitado"
            log_solicitacao(cpf, current_limit, requested_value, status)
            return (
                f"REJEITADO: Com o score atual de {score} pontos, o limite máximo "
                f"permitido é R$ {max_allowed:,.2f}.\n"
                f"Valor solicitado: R$ {requested_value:,.2f}.\n"
                f"RECOMENDAÇÃO: O cliente pode realizar uma entrevista de reavaliação "
                f"de crédito para melhorar o score e conseguir um limite maior."
            )
    except Exception as e:
        logger.error("request_limit_increase | cpf=%s | valor=%.2f | %s", cpf, requested_value, e, exc_info=True)
        return f"ERRO: Falha ao processar solicitação de aumento: {str(e)}"


# ============================================================
# Tool: Cálculo e Atualização de Score (Entrevista)
# ============================================================

# Pesos definidos pelo desafio técnico
PESO_RENDA = 30
PESO_EMPREGO = {
    "formal": 300,
    "clt": 300,
    "autônomo": 200,
    "autonomo": 200,
    "desempregado": 0,
}
PESO_DEPENDENTES = {
    0: 100,
    1: 80,
    2: 60,
}
PESO_DEPENDENTES_3_PLUS = 30
PESO_DIVIDAS = {
    "sim": -100,
    "não": 100,
    "nao": 100,
}


@tool
def calculate_and_update_score(cpf: str, monthly_income: float,
                                employment_type: str, monthly_expenses: float,
                                dependents: int, has_debts: str) -> str:
    """
    Calcula um novo score de crédito com base nos dados da entrevista
    e atualiza o cadastro do cliente em clientes.csv.

    Fórmula: score = (renda / (despesas + 1)) * peso_renda + peso_emprego + peso_dependentes + peso_dividas

    Args:
        cpf: CPF do cliente (apenas números).
        monthly_income: Renda mensal em reais.
        employment_type: Tipo de emprego (formal, autônomo, desempregado).
        monthly_expenses: Despesas fixas mensais em reais.
        dependents: Número de dependentes.
        has_debts: Se possui dívidas ativas ("sim" ou "não").

    Returns:
        Resultado da reavaliação com o novo score.
    """
    try:
        client = find_client(cpf)
        if client is None:
            return "ERRO: CPF não encontrado na base de clientes."

        old_score = int(client["score"])
        old_limit = float(client["current_limit"])

        # ---- Cálculo do score conforme fórmula do desafio ----
        # Componente 1: Renda / (Despesas + 1) * peso_renda
        income_component = (monthly_income / (monthly_expenses + 1)) * PESO_RENDA

        # Componente 2: Peso do tipo de emprego
        emp = employment_type.lower().strip()
        employment_component = PESO_EMPREGO.get(emp, 100)

        # Componente 3: Peso dos dependentes
        if dependents >= 3:
            dependents_component = PESO_DEPENDENTES_3_PLUS
        else:
            dependents_component = PESO_DEPENDENTES.get(dependents, 60)

        # Componente 4: Peso das dívidas
        debts_key = has_debts.lower().strip()
        debts_component = PESO_DIVIDAS.get(debts_key, 0)

        # Score final (clamp 0-1000)
        score = income_component + employment_component + dependents_component + debts_component
        score = max(0, min(1000, int(round(score))))

        # Consultar novo limite baseado no score atualizado
        new_limit = get_max_limit_for_score(score)
        # Garantir que o novo limite não seja menor que o atual
        new_limit = max(new_limit, old_limit)

        # Atualizar no clients.csv
        update_client(cpf, {
            "score": str(score),
            "current_limit": f"{new_limit:.2f}",
        })

        return (
            f"REAVALIAÇÃO CONCLUÍDA:\n"
            f"- Score anterior: {old_score} → Novo score: {score}\n"
            f"- Limite anterior: R$ {old_limit:,.2f} → Novo limite: R$ {new_limit:,.2f}\n"
            f"{'Parabéns! Seu score melhorou!' if score > old_score else 'Seu score foi mantido ou ajustado.'}"
        )
    except Exception as e:
        logger.error("calculate_and_update_score | cpf=%s | %s", cpf, e, exc_info=True)
        return f"ERRO: Falha ao calcular score: {str(e)}"


# ============================================================
# Tool: Consulta de Câmbio
# ============================================================

@tool
def get_currency_rate(currency: str) -> str:
    """
    Consulta a cotação atual de uma moeda estrangeira em relação ao Real (BRL).

    Args:
        currency: Código da moeda (ex: USD, EUR, GBP, JPY).

    Returns:
        Cotação atualizada da moeda em relação ao BRL.
    """
    currency = currency.upper().strip()

    try:
        url = f"https://api.frankfurter.app/latest?from={currency}&to=BRL"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "rates" in data and "BRL" in data["rates"]:
            rate = data["rates"]["BRL"]
            return (
                f"Cotação atual: 1 {currency} = R$ {rate:.4f}\n"
                f"Data da consulta: {data.get('date', 'N/A')}"
            )
        else:
            return f"Não foi possível obter a cotação para {currency}. Verifique o código da moeda."

    except requests.exceptions.RequestException as e:
        logger.error("get_currency_rate | currency=%s | %s", currency, e, exc_info=True)
        return f"Erro ao consultar a API de câmbio: {str(e)}"


# ============================================================
# Tool: Encerramento de Conversa
# ============================================================

@tool
def end_conversation() -> str:
    """
    Encerra o atendimento atual. Deve ser chamada quando o cliente
    deseja finalizar a conversa ou quando o atendimento é concluído.

    Returns:
        Mensagem confirmando o encerramento.
    """
    return "ENCERRAMENTO: Atendimento finalizado com sucesso."

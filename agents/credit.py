from langchain_core.messages import SystemMessage
from tools import check_limit, request_limit_increase, end_conversation
from agents.core import get_llm, _has_tool_result, _get_last_human_content, _make_tool_call_message
import re

def _extract_amount(text: str) -> float:
    """Extrai valor monetário de um texto."""
    cleaned = re.sub(r'[R$\s]', '', text)
    match = re.search(r'\d[\d.,]*', cleaned)
    if match:
        try:
            return float(match.group().replace('.', '').replace(',', '.'))
        except ValueError:
            pass
    return 0.0

CREDIT_SYSTEM_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil — especialista em crédito.
{name_line}

Ferramentas:
- check_limit: consulta o limite e score atual
- request_limit_increase: solicita aumento de limite

Fluxo:
1. Use check_limit para ver o limite e score atuais, se ainda não fez.
2. Informe os dados ao cliente (limite e score).
3. Se o cliente quiser AUMENTAR o limite: pergunte o valor desejado, depois chame request_limit_increase.
   - APROVADO: parabenize e informe o novo limite.
   - REJEITADO: explique o motivo, ofereça entrevista de reavaliação de score e aguarde resposta.
4. Se o cliente quiser apenas CONSULTAR (score, limite, saldo): informe os dados e pergunte se deseja mais algo.
   NÃO ofereça aumento de limite automaticamente — espere o cliente pedir.

Se o cliente quiser encerrar, chame end_conversation.
Tom: respeitoso, didático. Português do Brasil."""

def credit_agent(state: dict) -> dict:
    messages = state["messages"]
    cpf      = state.get("current_user_cpf", "")

    check_done   = _has_tool_result(messages, "Dados de crédito")
    increase_done = (
        _has_tool_result(messages, "APROVADO") or
        _has_tool_result(messages, "REJEITADO")
    )

    if not increase_done:
        last_human   = _get_last_human_content(messages)
        amount       = _extract_amount(last_human)
        increase_kws = ["aumentar", "aumento", "quero", "gostaria", "preciso",
                        "solicitar", "limite de", "novo limite"]

        if amount > 0 and any(w in last_human.lower() for w in increase_kws):
            return {
                "messages": [_make_tool_call_message(
                    "request_limit_increase", {"cpf": cpf, "requested_value": amount}
                )],
                "current_agent": "credit",
                "routing_target": "",
            }

        if not check_done:
            return {
                "messages": [_make_tool_call_message("check_limit", {"cpf": cpf})],
                "current_agent": "credit",
                "routing_target": "",
            }

    llm = get_llm()
    llm_with_tools = llm.bind_tools([check_limit, request_limit_increase, end_conversation])

    user_name     = state.get("current_user_name", "")
    name_line     = f"Cliente: {user_name}" if user_name else ""
    system_prompt = CREDIT_SYSTEM_PROMPT.format(name_line=name_line)

    msgs     = [SystemMessage(content=system_prompt)] + messages
    response = llm_with_tools.invoke(msgs)

    return {"messages": [response], "current_agent": "credit", "routing_target": ""}

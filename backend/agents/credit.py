from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import tool
from tools import check_limit, request_limit_increase, end_conversation
from agents.core import get_llm, _has_tool_result, _get_last_human_content, _make_tool_call_message, _invoke_with_retry, _trim_messages
import re

@tool
def transfer_to_interview() -> str:
    """Aciona a transferência do cliente para a Entrevista Financeira Rápida. USE APENAS SE O CLIENTE ACEITAR a oferta de entrevista explicitamente."""
    return "INICIANDO_ENTREVISTA_FINANCEIRA"

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

CREDIT_SYSTEM_PROMPT = """Você é a Bia, gerente de crédito do Banco Ágil.
{name_line}
CPF do cliente: {cpf}

Ferramentas disponíveis: {tools_list}

Regras:
{check_limit_rule}- Aumento solicitado: chame request_limit_increase com cpf="{cpf}" e o valor informado pelo cliente.
  APROVADO → comemore com entusiasmo real, seja genuíno.
  REJEITADO → seja empático e humano. Ofereça a entrevista de reavaliação como uma oportunidade, não como um roteiro.
- Cliente aceitar entrevista → chame transfer_to_interview.
- Encerrar → end_conversation.

Tom: Fale como uma pessoa real, não como um atendente de script. Varie as respostas.
Não termine sempre com a mesma pergunta genérica. Português do Brasil."""

def credit_agent(state: dict) -> dict:
    messages = state["messages"]
    cpf      = state.get("current_user_cpf", "")

    check_done    = _has_tool_result(messages, "Dados de crédito")
    reaval_done   = _has_tool_result(messages, "REAVALIAÇÃO CONCLUÍDA")
    # Após reavaliação de score, reseta o gate de aumento para permitir novo pedido
    increase_done = (
        _has_tool_result(messages, "APROVADO") or
        (_has_tool_result(messages, "REJEITADO") and not reaval_done)
    )

    if _has_tool_result(messages, "INICIANDO_ENTREVISTA_FINANCEIRA"):
        return {
            "messages": [AIMessage(content="Perfeito! Vou chamar a nossa plataforma de testes, um momento... 🚀")],
            "current_agent": "credit",
            "routing_target": "interview",
        }

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
    # Não oferece check_limit ao LLM se já foi consultado — evita re-chamada com CPF errado
    available_tools = [request_limit_increase, end_conversation, transfer_to_interview]
    if not check_done:
        available_tools.insert(0, check_limit)
    llm_with_tools = llm.bind_tools(available_tools)

    user_name  = state.get("current_user_name", "")
    name_line  = f"Cliente: {user_name}" if user_name else ""
    tools_list = " | ".join(t.name for t in available_tools)
    check_limit_rule = (
        f'- Limite/score: chame check_limit com cpf="{cpf}". NUNCA invente valores.\n'
        if not check_done else ""
    )
    system_prompt = CREDIT_SYSTEM_PROMPT.format(
        name_line=name_line, cpf=cpf,
        tools_list=tools_list, check_limit_rule=check_limit_rule,
    )

    msgs     = [SystemMessage(content=system_prompt)] + _trim_messages(messages)
    response = _invoke_with_retry(llm_with_tools, msgs)

    return {"messages": [response], "current_agent": "credit", "routing_target": ""}

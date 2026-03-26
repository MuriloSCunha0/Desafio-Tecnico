from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from tools import get_currency_rate, end_conversation
from agents.core import get_llm, _normalize_content, _make_tool_call_message, _invoke_with_retry, _trim_messages

FOREX_SYSTEM_PROMPT = """Você é o Bia, especialista de câmbio do Banco Ágil.
{name_line}

Ferramentas: get_currency_rate | end_conversation
Traduza: dólar→USD, euro→EUR, libra→GBP, iene→JPY.

REGRA CRÍTICA: NUNCA informe cotações sem chamar get_currency_rate primeiro.
Após a cotação, mostre o valor de forma clara e natural — como um amigo que entende de câmbio.
Se pedir conversão, calcule e explique rapidinho. Varie o jeito de responder.
Encerrar → end_conversation.

Português do Brasil. Seja leve e direto, sem parecer robô."""

def _detect_currency(messages: list) -> str:
    def _currency_from_text(text: str) -> str:
        if any(w in text for w in ["dólar", "dolar", "usd", "dollar"]):
            return "USD"
        if any(w in text for w in ["euro", "eur"]):
            return "EUR"
        if any(w in text for w in ["libra", "gbp"]):
            return "GBP"
        if any(w in text for w in ["iene", "jpy", "yen"]):
            return "JPY"
        return ""

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            found = _currency_from_text(str(msg.content).lower())
            if found:
                return found
            affirmatives = ["sim", "ok", "pode", "claro", "isso", "esse", "essa", "quero"]
            if any(w in str(msg.content).lower() for w in affirmatives):
                for prev in reversed(messages):
                    if isinstance(prev, AIMessage) and prev.content:
                        found = _currency_from_text(_normalize_content(prev.content).lower())
                        if found:
                            return found
                        break
            break
    return ""

def _has_currency_result(messages: list, currency: str) -> bool:
    return any(
        isinstance(m, ToolMessage) and f"1 {currency}" in str(m.content)
        for m in messages
    )

def forex_agent(state: dict) -> dict:
    messages = state["messages"]

    currency = _detect_currency(messages)
    if currency and not _has_currency_result(messages, currency):
        return {
            "messages": [_make_tool_call_message(
                "get_currency_rate", {"currency": currency}
            )],
            "current_agent": "forex",
            "routing_target": "",
        }

    llm = get_llm()
    llm_with_tools = llm.bind_tools([get_currency_rate, end_conversation])

    user_name     = state.get("current_user_name", "")
    name_line     = f"Cliente: {user_name}" if user_name else ""
    system_prompt = FOREX_SYSTEM_PROMPT.format(name_line=name_line)

    msgs     = [SystemMessage(content=system_prompt)] + _trim_messages(messages)
    response = _invoke_with_retry(llm_with_tools, msgs)

    return {"messages": [response], "current_agent": "forex", "routing_target": ""}

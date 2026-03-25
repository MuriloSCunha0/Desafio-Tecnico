from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from tools import get_currency_rate, end_conversation
from agents.core import get_llm, _normalize_content, _make_tool_call_message

FOREX_SYSTEM_PROMPT = """Você é o Agilito, especialista em moedas e câmbio do Banco Ágil.
{name_line}

Use get_currency_rate para buscar as cotações em tempo real. Traduza: dólar→USD, euro→EUR, libra→GBP, iene→JPY.
Após devolver a cotação com a taxa do dia, pergunte de forma simpática se o cliente quer fazer alguma conversão ou se precisa de mais algo.
Se o cliente quiser converter um valor (ex: "quanto dá 100 dólares"), faça a multiplicação de forma clara, explicando o cálculo rapidinho.
Para encerrar, use end_conversation.

Tom: Dinâmico, prestativo e natural 🚀. Respostas curtas, sem enrolação. Português do Brasil."""

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

    msgs     = [SystemMessage(content=system_prompt)] + messages
    response = llm_with_tools.invoke(msgs)

    return {"messages": [response], "current_agent": "forex", "routing_target": ""}

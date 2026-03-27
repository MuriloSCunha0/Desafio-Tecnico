import re
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from tools import get_currency_rate, convert_currency, end_conversation
from agents.core import get_llm, get_llm_fast, _normalize_content, _make_tool_call_message, _invoke_with_retry, _trim_messages, _strip_llm_artifacts

FOREX_SYSTEM_PROMPT = """Você é o Bia, especialista de câmbio do Banco Ágil.
{name_line}

Ferramentas: get_currency_rate | convert_currency | end_conversation
Traduza as moedas mais comuns (ex: dólar→USD, euro→EUR, libra→GBP, iene→JPY, real→BRL).
Para TODAS as outras moedas do mundo, utilize o código ISO 4217 padrão (ex: franco suíço→CHF, peso argentino→ARS, dólar canadense→CAD, etc).

Regras:
- Cotação de uma moeda em BRL → chame get_currency_rate(currency).
- Conversão entre duas moedas (ex: "100 dólares em euros") → chame convert_currency(from_currency, to_currency, amount).
- NUNCA informe valores sem chamar a ferramenta adequada primeiro.
- NUNCA inclua JSON ou código na resposta de texto.
- Após o resultado, explique de forma clara e natural. Varie o jeito de responder.
- Encerrar → end_conversation.

Português do Brasil. Seja leve e direto, sem parecer robô."""


_CURRENCY_KEYWORDS = [
    (["dólar", "dolar", "usd", "dollar"], "USD"),
    (["euro", "eur"], "EUR"),
    (["libra", "gbp"], "GBP"),
    (["iene", "jpy", "yen"], "JPY"),
    (["real", "brl", "reais"], "BRL"),
]


def _currency_from_text(text: str) -> str:
    """Retorna o código da primeira moeda estrangeira encontrada no texto (ignora BRL)."""
    for keywords, code in _CURRENCY_KEYWORDS:
        if code == "BRL":
            continue
        if any(w in text for w in keywords):
            return code
    return ""


def _detect_conversion(messages: list) -> tuple[str, str, float]:
    """Detecta pedido de conversão entre duas moedas com valor.
    A ordem de from/to é determinada pela posição no texto, não pela lista de keywords.
    Ex: '100 euros em dólar' → EUR→USD (correto).
    """
    for msg in reversed(messages):
        if not isinstance(msg, HumanMessage):
            continue
        text = str(msg.content).lower()

        # Encontra a posição mais cedo de cada moeda no texto
        currency_positions: list[tuple[int, str]] = []
        for keywords, code in _CURRENCY_KEYWORDS:
            best_pos = None
            for kw in keywords:
                idx = text.find(kw)
                if idx != -1 and (best_pos is None or idx < best_pos):
                    best_pos = idx
            if best_pos is not None:
                currency_positions.append((best_pos, code))

        # Ordena pela posição no texto → preserva from/to correto
        currency_positions.sort(key=lambda x: x[0])
        found = [code for _, code in currency_positions]

        if len(found) >= 2:
            match = re.search(r'(\d[\d.,]*)', text)
            amount = 1.0
            if match:
                try:
                    amount = float(match.group(1).replace(",", "."))
                except ValueError:
                    pass
            return found[0], found[1], amount
        break
    return "", "", 0.0


def _detect_currency(messages: list) -> str:
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


def _has_conversion_result(messages: list, from_c: str, to_c: str) -> bool:
    return any(
        isinstance(m, ToolMessage) and f"{from_c}" in str(m.content) and f"{to_c}" in str(m.content)
        for m in messages
    )


def forex_agent(state: dict) -> dict:
    messages = state["messages"]

    # Conversão entre duas moedas — tem prioridade sobre cotação simples
    from_c, to_c, amount = _detect_conversion(messages)
    if from_c and to_c and not _has_conversion_result(messages, from_c, to_c):
        return {
            "messages": [_make_tool_call_message(
                "convert_currency", {"from_currency": from_c, "to_currency": to_c, "amount": amount}
            )],
            "current_agent": "forex",
            "routing_target": "",
        }

    # Cotação simples de uma moeda em BRL
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
    llm_with_tools = llm.bind_tools([get_currency_rate, convert_currency, end_conversation])

    user_name     = state.get("current_user_name", "")
    name_line     = f"Cliente: {user_name}" if user_name else ""
    system_prompt = FOREX_SYSTEM_PROMPT.format(name_line=name_line)

    msgs     = [SystemMessage(content=system_prompt)] + _trim_messages(messages)
    response = _invoke_with_retry(llm_with_tools, msgs)

    # Remove artefatos Groq (JSON / <function=...> vazados no texto)
    cleaned = _strip_llm_artifacts(response.content)
    if cleaned != (response.content or ""):
        response = AIMessage(content=cleaned, tool_calls=[], additional_kwargs={})

    return {"messages": [response], "current_agent": "forex", "routing_target": ""}

import os
import re
import time
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

def _invoke_with_retry(llm, msgs, max_retries: int = 4):
    """Invoca o LLM com retry automático para Rate Limit (429) do Groq/Google."""
    delay = 5
    for attempt in range(max_retries):
        try:
            return llm.invoke(msgs)
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower() or "Rate limit" in err
            if is_rate_limit and attempt < max_retries - 1:
                wait = delay * (2 ** attempt)  # 5s, 10s, 20s, 40s
                time.sleep(wait)
                continue
            raise

def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            convert_system_message_to_human=True,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.3,
        )

def _normalize_content(content) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content) if content else ""

def _extract_cpf(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            cleaned = re.sub(r'[.\-\s]', '', str(msg.content))
            match = re.search(r'\d{11}', cleaned)
            if match:
                return match.group()
    return ""

def _extract_dob_from_last_human(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            match = re.search(
                r'\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4}|\d{2})\b',
                str(msg.content)
            )
            if match:
                return match.group()
    return ""

def _get_last_human_content(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""

def _last_msg_is_human(messages: list) -> bool:
    return bool(messages) and isinstance(messages[-1], HumanMessage)

def _last_msg_is_tool(messages: list) -> bool:
    return bool(messages) and isinstance(messages[-1], ToolMessage)

def _detect_routing_target(user_msg: str, messages: list = None) -> str:
    msg = user_msg.lower()

    if any(w in msg for w in ["encerrar", "encerra", "finalizar", "finaliza",
                               "tchau", "até logo", "ate logo", "sair", "fechar"]):
        return ""

    if any(w in msg for w in ["entrevista", "reavalia", "melhorar score",
                               "melhorar meu score", "aumentar score",
                               "reavaliação", "reavaliar",
                               "entrevista de crédito", "entrevista financeira"]):
        return "interview"

    if any(w in msg for w in ["limite", "crédito", "credito", "aumento", "saldo"]):
        return "credit"

    if any(w in msg for w in ["câmbio", "cambio", "dólar", "dolar", "euro",
                               "libra", "iene", "moeda", "cotação", "cotacao"]):
        return "forex"

    if any(w in msg for w in ["score", "pontuação", "pontuacao"]):
        return "credit"

    affirmatives = ["sim", "ok", "pode", "claro", "aceito", "topo", "quero", "vamos", "gostaria"]
    if messages and any(w in msg for w in affirmatives):
        for prev_msg in reversed(messages):
            if isinstance(prev_msg, AIMessage) and prev_msg.content:
                prev_text = _normalize_content(prev_msg.content).lower()
                if "entrevista" in prev_text or "reavalia" in prev_text:
                    return "interview"
                if "câmbio" in prev_text or "cotação" in prev_text or "moeda" in prev_text:
                    return "forex"
                if "limite" in prev_text or "crédito" in prev_text:
                    return "credit"
                break

    return ""

def _make_tool_call_message(tool_name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "id": str(uuid.uuid4()),
            "name": tool_name,
            "args": args,
        }],
    )

def _trim_messages(messages: list, max_messages: int = 4) -> list:
    """Retorna apenas as últimas N mensagens para enviar ao LLM.
    Sempre preserva ToolMessages para não quebrar o protocolo tool_call→tool_result."""
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]

def _has_tool_result(messages: list, substring: str) -> bool:
    return any(
        isinstance(m, ToolMessage) and substring in str(m.content)
        for m in messages
    )

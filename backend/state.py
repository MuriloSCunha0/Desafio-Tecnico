"""
state.py — Definição do estado e do grafo multi-agente do Banco Ágil.

Contém:
- AgentState (TypedDict) com todos os campos de controle
- Construção do StateGraph com nós e arestas condicionais
- SqliteSaver para persistência de conversas
- Nó de execução de tools
"""

import json
import os
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessage, ToolMessage

from tools import (
    authenticate_user,
    check_limit,
    request_limit_increase,
    calculate_and_update_score,
    get_currency_rate,
    end_conversation,
)


# ============================================================
# Definição do Estado
# ============================================================

class AgentState(TypedDict):
    """Estado compartilhado entre todos os agentes do Banco Ágil."""
    messages: Annotated[list, add_messages]
    is_authenticated: bool
    auth_attempts: int
    current_user_cpf: str
    current_user_name: str
    pending_cpf: str          # CPF coletado antes da autenticação ser confirmada
    routing_target: str       # Alvo de roteamento definido pelo triage via Python
    interview_context: dict
    current_agent: str


# ============================================================
# Mapeamento de ferramentas
# ============================================================

TOOLS_MAP = {
    "authenticate_user": authenticate_user,
    "check_limit": check_limit,
    "request_limit_increase": request_limit_increase,
    "calculate_and_update_score": calculate_and_update_score,
    "get_currency_rate": get_currency_rate,
    "end_conversation": end_conversation,
}


# ============================================================
# Nó de execução de tools
# ============================================================

def tool_executor(state: AgentState) -> dict:
    """Executa as tool calls pendentes na última mensagem do agente."""
    last_message = state["messages"][-1]
    results = []
    state_updates = {}

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_fn = TOOLS_MAP.get(tool_name)

        if tool_fn is None:
            result = f"Erro: ferramenta '{tool_name}' não encontrada."
        else:
            try:
                result = tool_fn.invoke(tool_args)
            except Exception as e:
                result = f"Erro ao executar a ferramenta '{tool_name}': {str(e)}"

        results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

        # Processar efeitos colaterais no estado
        if tool_name == "authenticate_user":
            if "SUCESSO" in str(result):
                cpf = tool_args.get("cpf", "")
                # Extrair nome da mensagem de sucesso (formato: "SUCESSO: Usuário autenticado! Bem-vindo(a), {name}!")
                result_str = str(result)
                name_start = result_str.find("Bem-vindo(a), ") + len("Bem-vindo(a), ")
                name_end = result_str.find("!", name_start)
                user_name = result_str[name_start:name_end] if name_start > 0 and name_end > 0 else ""
                
                state_updates["is_authenticated"] = True
                state_updates["current_user_cpf"] = cpf
                state_updates["current_user_name"] = user_name
                state_updates["auth_attempts"] = 0
                state_updates["pending_cpf"] = ""  # limpar CPF temporário
            else:
                current_attempts = state.get("auth_attempts", 0) + 1
                state_updates["auth_attempts"] = current_attempts

        elif tool_name == "end_conversation":
            state_updates["current_agent"] = "ended"
            # Gera despedida personalizada com o nome do cliente
            user_name = state.get("current_user_name", "")
            first_name = user_name.split()[0] if user_name else ""
            farewell_options = [
                f"Foi um prazer te atender{', ' + first_name if first_name else ''}! 😊 Se precisar de qualquer coisa, é só voltar. Até logo!",
                f"Encerrando por aqui{', ' + first_name if first_name else ''}! 🙌 Qualquer dúvida, estamos sempre à disposição. Cuide-se!",
                f"Tudo certo{', ' + first_name if first_name else ''}! 👋 Foi ótimo ajudar você hoje. Conte com o Banco Ágil sempre que precisar!",
            ]
            import random
            farewell = random.choice(farewell_options)
            results.append(AIMessage(content=farewell))

    return {"messages": results, **state_updates}


# ============================================================
# Roteadores (funções de decisão do grafo)
# ============================================================

def should_use_tools(state: AgentState) -> str:
    """Decide se o agente quer chamar uma tool ou se já respondeu."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "respond"


def route_after_tools(state: AgentState) -> str:
    """Após executar tools, volta para o agente atual continuar a conversa."""
    current = state.get("current_agent", "triage")

    # Se bloqueado ou encerrado, encerra
    if current in ("blocked", "ended"):
        return END

    return current


def route_after_response(state: AgentState) -> str:
    """Após o agente responder (sem tool calls), decide a próxima ação."""
    last_message = state["messages"][-1]
    content = last_message.content if isinstance(last_message, AIMessage) else ""
    # Google Gemini pode retornar content como lista de dicts; extrair texto.
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
            else:
                parts.append(str(part))
        content = "\n".join(parts)
    content = str(content) if content else ""

    # Verificar bloqueio
    if state.get("auth_attempts", 0) >= 3:
        return END

    current = state.get("current_agent", "triage")

    # Se bloqueado ou encerrado, encerra
    if current in ("blocked", "ended"):
        return END

    # 1. Verificar routing_target definido pelo triage via Python (mais confiável)
    routing_target = state.get("routing_target", "")
    if routing_target == "credit":
        return "credit"
    if routing_target == "interview":
        return "interview"
    if routing_target == "forex":
        return "forex"

    # 2. Fallback: detectar ROTA: tags no conteúdo (para roteamento inter-agentes)
    content_upper = content.upper()
    if "ROTA:CREDITO" in content_upper:
        return "credit"
    if "ROTA:ENTREVISTA" in content_upper:
        return "interview"
    if "ROTA:CAMBIO" in content_upper:
        return "forex"
    if "ROTA:TRIAGEM" in content_upper:
        return "triage"

    # Sem roteamento — esperar próxima mensagem do usuário
    return END


# ============================================================
# Construção do Grafo
# ============================================================

def build_graph(db_path: str = None, use_memory: bool = False):
    """
    Constrói e retorna o grafo compilado do sistema multi-agente
    com persistência via SqliteSaver ou MemorySaver.

    Args:
        db_path: Caminho para o banco SQLite de persistência.
                 Se None, usa diretório temporário do sistema (evita OneDrive).
        use_memory: Se True, usa MemorySaver (sem disco, ideal para testes).

    Returns:
        Tuple (compiled_graph, connection_or_none) — manter a conexão aberta
        enquanto o grafo estiver em uso.
    """
    import sqlite3

    from agents import triage_agent, credit_agent, interview_agent, forex_agent
    from agents.core import _get_last_human_content, _detect_routing_target

    # ── Roteador de entrada inteligente (sem custo LLM) ─────────
    def smart_entry(state: AgentState) -> dict:
        """Roteador zero-cost: se o usuário já está em um agente
        especialista, pula o triage e vai direto — economiza 1 LLM call."""
        return {}  # Não altera estado, só decide rota

    def route_entry(state: AgentState) -> str:
        """Decide para qual agente enviar: triage ou direto ao especialista."""
        current = state.get("current_agent", "triage")
        is_auth = state.get("is_authenticated", False)
        messages = state.get("messages", [])

        # Não autenticado → triage sempre
        if not is_auth:
            return "triage"

        # Se já está em um agente especialista, verifica se o usuário quer trocar
        if current in ("credit", "interview", "forex"):
            user_msg = _get_last_human_content(messages)
            new_target = _detect_routing_target(user_msg, messages)

            # Encerrar → triage para processar
            encerrar_words = ["encerrar", "encerra", "tchau", "sair", "finalizar"]
            if any(w in user_msg.lower() for w in encerrar_words):
                return "triage"

            # Usuário quer trocar de agente
            if new_target and new_target != current:
                return new_target

            # Continua no agente atual (sem triage!)
            return current

        # Triage ou estado desconhecido
        return "triage"

    # Construir o grafo
    graph = StateGraph(AgentState)

    # Adicionar nós dos agentes
    graph.add_node("smart_entry", smart_entry)
    graph.add_node("triage", triage_agent)
    graph.add_node("credit", credit_agent)
    graph.add_node("interview", interview_agent)
    graph.add_node("forex", forex_agent)
    graph.add_node("tools", tool_executor)

    # Ponto de entrada: smart_entry (zero-cost router)
    graph.set_entry_point("smart_entry")
    graph.add_conditional_edges(
        "smart_entry",
        route_entry,
        {
            "triage": "triage",
            "credit": "credit",
            "interview": "interview",
            "forex": "forex",
        },
    )

    # Arestas condicionais — cada agente decide: chamar tool ou responder?
    for agent_name in ["triage", "credit", "interview", "forex"]:
        graph.add_conditional_edges(
            agent_name,
            should_use_tools,
            {
                "tools": "tools",
                "respond": "route_response",
            },
        )

    # Nó virtual de roteamento pós-resposta
    graph.add_node("route_response", lambda state: {})
    graph.add_conditional_edges(
        "route_response",
        route_after_response,
        {
            "triage": "triage",
            "credit": "credit",
            "interview": "interview",
            "forex": "forex",
            END: END,
        },
    )

    # Após executar tools, volta para o agente atual
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "triage": "triage",
            "credit": "credit",
            "interview": "interview",
            "forex": "forex",
            END: END,
        },
    )

    # Checkpointer
    conn = None
    if use_memory:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
    else:
        # SQLite — usa diretório temp para evitar locks do OneDrive
        if db_path is None:
            import tempfile
            db_path = os.path.join(tempfile.gettempdir(), "banco_agil.db")

        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA wal_autocheckpoint=100")
        checkpointer = SqliteSaver(conn)

    compiled = graph.compile(checkpointer=checkpointer)

    return compiled, conn


"""
app.py — Frontend Streamlit do Banco Ágil.

Interface de chat que se conecta ao backend LangGraph,
simulando um atendimento bancário contínuo.
"""

import uuid
import re
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from state import build_graph


def _normalize_content(content) -> str:
    """Normaliza o content de uma AIMessage para string.

    Google Gemini pode retornar content como lista de dicts, ex:
    [{'type': 'text', 'text': '...', 'extras': {...}}].
    Esta função extrai apenas o texto legível.
    """
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


def sanitize_internal_tags(content) -> str:
    """Remove tags internas de roteamento antes de exibir ao cliente."""
    content = _normalize_content(content)
    cleaned = re.sub(r"\bROTA:[A-Z_]+\b", "", content)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ============================================================
# Configuração da página
# ============================================================

st.set_page_config(
    page_title="Banco Ágil — Atendimento Inteligente",
    page_icon="🏦",
    layout="centered",
)

# ============================================================
# CSS customizado
# ============================================================

st.markdown("""
<style>
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a5276, #2e86c1);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .status-authenticated {
        background: #27ae60;
        color: white;
    }
    .status-pending {
        background: #f39c12;
        color: white;
    }
    .status-blocked {
        background: #e74c3c;
        color: white;
    }

    /* Chat container */
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Inicialização do estado da sessão
# ============================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "graph" not in st.session_state:
    graph, conn = build_graph()
    st.session_state.graph = graph
    st.session_state.db_conn = conn

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "is_authenticated": False,
        "auth_attempts": 0,
        "current_user_cpf": "",
        "current_user_name": "",
        "pending_cpf": "",
        "routing_target": "",
        "current_agent": "triage",
    }

if not st.session_state.chat_history:
    initial_message = (
        "Oi! Eu sou o Agilito, assistente virtual do Banco Ágil. 😊\n\n"
        "Posso te ajudar com:\n"
        "- consulta de limite de crédito\n"
        "- solicitação de aumento de limite\n"
        "- reavaliação de score por entrevista\n"
        "- cotação de moedas\n\n"
        "Para sua segurança, vamos começar com uma autenticação rápida.\n"
        "Pode me informar seu CPF (somente números)?"
    )
    st.session_state.chat_history.append({"role": "assistant", "content": initial_message})

    # Seed the graph checkpoint with the welcome message so the LLM
    # knows the greeting already happened and won't repeat the CPF request.
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    st.session_state.graph.update_state(
        config,
        {
            "messages": [AIMessage(content=initial_message)],
            "is_authenticated": False,
            "auth_attempts": 0,
            "current_user_cpf": "",
            "current_user_name": "",
            "pending_cpf": "",
            "routing_target": "",
            "interview_context": {},
            "current_agent": "triage",
        },
    )


# ============================================================
# Header
# ============================================================

auth_state = st.session_state.agent_state
if auth_state.get("auth_attempts", 0) >= 3:
    status_class = "status-blocked"
    status_text = "🔒 Conta Bloqueada"
elif auth_state.get("is_authenticated"):
    status_class = "status-authenticated"
    status_text = "✅ Autenticado"
else:
    status_class = "status-pending"
    status_text = "⏳ Aguardando Autenticação"

st.markdown(f"""
<div class="main-header">
    <h1>🏦 Banco Ágil</h1>
    <p>Seu atendimento bancário inteligente e acessível</p>
    <span class="status-badge {status_class}">{status_text}</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Exibir histórico do chat
# ============================================================

for msg in st.session_state.chat_history:
    role = msg["role"]
    content = msg["content"]
    with st.chat_message(role):
        st.markdown(content)


# ============================================================
# Input do usuário
# ============================================================

if user_input := st.chat_input("Digite sua mensagem..."):
    # Exibir mensagem do usuário
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Verificar se a conta está bloqueada
    if st.session_state.agent_state.get("auth_attempts", 0) >= 3:
        blocked_msg = (
            "🔒 Sua conta está bloqueada. Por favor, procure a agência "
            "mais próxima do Banco Ágil com um documento de identidade."
        )
        st.session_state.chat_history.append({"role": "assistant", "content": blocked_msg})
        with st.chat_message("assistant"):
            st.markdown(blocked_msg)
        st.rerun()

    # Invocar o grafo
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        with st.spinner("Processando..."):
            try:
                result = st.session_state.graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "is_authenticated": st.session_state.agent_state.get("is_authenticated", False),
                        "auth_attempts": st.session_state.agent_state.get("auth_attempts", 0),
                        "current_user_cpf": st.session_state.agent_state.get("current_user_cpf", ""),
                        "current_user_name": st.session_state.agent_state.get("current_user_name", ""),
                        "pending_cpf": st.session_state.agent_state.get("pending_cpf", ""),
                        "routing_target": st.session_state.agent_state.get("routing_target", ""),
                        "interview_context": st.session_state.agent_state.get("interview_context", {}),
                        "current_agent": st.session_state.agent_state.get("current_agent", "triage"),
                    },
                    config=config,
                )

                # Atualizar estado local
                st.session_state.agent_state["is_authenticated"] = result.get("is_authenticated", False)
                st.session_state.agent_state["auth_attempts"] = result.get("auth_attempts", 0)
                st.session_state.agent_state["current_user_cpf"] = result.get("current_user_cpf", "")
                st.session_state.agent_state["current_user_name"] = result.get("current_user_name", "")
                st.session_state.agent_state["pending_cpf"] = result.get("pending_cpf", "")
                st.session_state.agent_state["routing_target"] = result.get("routing_target", "")
                st.session_state.agent_state["current_agent"] = result.get("current_agent", "triage")
                if "interview_context" in result:
                    st.session_state.agent_state["interview_context"] = result["interview_context"]

                # Extrair a última resposta do agente (ignorar ToolMessages)
                ai_messages = [
                    m for m in result["messages"]
                    if isinstance(m, AIMessage) and m.content and not m.tool_calls
                ]

                if ai_messages:
                    last_response = ai_messages[-1].content
                    last_response = sanitize_internal_tags(last_response)

                    st.markdown(last_response)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": last_response}
                    )
                else:
                    fallback = "Desculpe, não consegui processar sua solicitação. Pode tentar novamente?"
                    st.markdown(fallback)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": fallback}
                    )

            except Exception as e:
                error_msg = f"⚠️ Ocorreu um erro inesperado: {str(e)}"
                st.markdown(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )

    st.rerun()


# ============================================================
# Sidebar com informações
# ============================================================

with st.sidebar:
    st.markdown("### ℹ️ Sobre o Banco Ágil")
    st.markdown(
        "O Banco Ágil oferece um atendimento inteligente, "
        "personalizado e acessível para facilitar sua vida financeira."
    )

    st.markdown("---")
    st.markdown("### 🤖 Serviços Disponíveis")
    st.markdown(
        "- 🔐 **Autenticação** — CPF e data de nascimento\n"
        "- 💳 **Crédito** — Consulta e aumento de limite\n"
        "- 📊 **Entrevista** — Reavaliação de score\n"
        "- 💱 **Câmbio** — Cotações de moedas"
    )

    st.markdown("---")
    st.markdown("### 📋 Dados da Sessão")
    st.markdown(f"**Thread ID:** `{st.session_state.thread_id[:8]}...`")

    if st.session_state.agent_state.get("is_authenticated"):
        st.markdown(f"**CPF:** `{st.session_state.agent_state.get('current_user_cpf', 'N/A')}`")
        if st.session_state.agent_state.get("current_user_name"):
            st.markdown(f"**Cliente:** `{st.session_state.agent_state.get('current_user_name')}`")

    # Agente ativo
    agent_labels = {
        "triage": "🔐 Triagem",
        "credit": "💳 Crédito",
        "interview": "📊 Entrevista",
        "forex": "💱 Câmbio",
        "ended": "✅ Encerrado",
        "blocked": "🔒 Bloqueado",
    }
    current_agent = st.session_state.agent_state.get("current_agent", "triage")
    agent_label = agent_labels.get(current_agent, f"❓ {current_agent}")
    st.markdown(f"**Agente Ativo:** {agent_label}")

    st.markdown("---")
    if st.button("🔄 Nova Conversa", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.agent_state = {
            "is_authenticated": False,
            "auth_attempts": 0,
            "current_user_cpf": "",
            "current_user_name": "",
            "pending_cpf": "",
            "routing_target": "",
            "interview_context": {},
            "current_agent": "triage",
        }
        st.rerun()

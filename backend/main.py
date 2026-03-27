from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import sys
import os
import asyncio
import logging

# Adiciona o diretório atual ao path para importação dos agentes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from state import build_graph
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional

logger = logging.getLogger(__name__)

# Definição dos modelos de entrada e saída
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None  # Enviado pelo frontend (pode ser temporário)
    cpf: Optional[str] = None        # Opcional, usado se já estiver logado no front

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    is_authenticated: bool
    current_agent: str
    cpf: str = None
    name: str = None

app = FastAPI(title="Banco Ágil API (LangGraph)")

# Configuração de CORS para permitir que o Frontend no Django chame a API REST
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instância Global do Grafo e conexão do banco de dados SQLite
# DB agora fica em %TEMP% para evitar locks do OneDrive
graph_instance, db_conn = build_graph()

# Timeout máximo para invoke do LangGraph (segundos)
INVOKE_TIMEOUT = 60

def sanitize_internal_tags(content: str) -> str:
    """Limpa marcações internas de roteamento do LLM (Ex: ROTA:CREDITO)."""
    import re
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
    else:
        content = str(content) if content else ""
    cleaned = re.sub(r"\bROTA:[A-Z_]+\b", "", content)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Endpoint de processamento centralizado.
    Se o cliente ainda não for autenticado, usa o thread_id anonimo.
    Se autenticado, o thread_id é ancorado no CPF automaticamente pelo Thread Hopping.
    """
    active_thread_id = req.cpf if req.cpf else (req.thread_id or str(uuid.uuid4()))
    config = {"configurable": {"thread_id": active_thread_id}}
    
    # Busca o estado atual para saber se a flag is_authenticated vai mudar neste turno
    current_state = graph_instance.get_state(config)
    was_authenticated = False
    if current_state and hasattr(current_state, "values") and current_state.values:
        was_authenticated = current_state.values.get("is_authenticated", False)

    try:
        # Envia apena a mensagem nova; o LangGraph gerencia o append no array `messages` via Reducer.
        input_data = {"messages": [HumanMessage(content=req.message)]}
        
        # Invoke com timeout para evitar travamento em rate-limit ou lock de SQLite
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: graph_instance.invoke(input_data, config=config)),
            timeout=INVOKE_TIMEOUT,
        )
        
        is_now_authenticated = result.get("is_authenticated", False)
        new_cpf = result.get("current_user_cpf", "")
        name = result.get("current_user_name", "")
        
        # ==============================================================
        # THREAD HOPPING LOGIC (Refatorado do Streamlit direto para a API)
        # ==============================================================
        if not was_authenticated and is_now_authenticated and new_cpf:
            try:
                cpf_config = {"configurable": {"thread_id": new_cpf}}
                cpf_state = graph_instance.get_state(cpf_config)
                
                # Se já exisita histórico salvo em disco SQLite no CPF
                if cpf_state and hasattr(cpf_state, "values") and cpf_state.values and len(cpf_state.values.get("messages", [])) > 0:
                    welcome_msg = f"Bem-vindo de volta, {name}! 🥳 Recuperei o histórico da nossa última conversa. Como posso ajudar hoje?"
                    
                    # Tenta injetar a mensagem. Usa as_node para evitar langgraph Errors
                    valid_nodes = {"triage", "credit", "interview", "forex"}
                    node_name = cpf_state.values.get("current_agent", "triage")
                    if node_name not in valid_nodes:
                        node_name = "triage"
                    graph_instance.update_state(cpf_config, {"messages": [AIMessage(content=welcome_msg)]}, as_node=node_name)
                    
                    return ChatResponse(
                        response=welcome_msg,
                        thread_id=new_cpf,
                        is_authenticated=True,
                        current_agent=cpf_state.values.get("current_agent", "triage"),
                        cpf=new_cpf,
                        name=name
                    )
                else:
                    # É a primeira vez do cliente — migra estado completo para a thread CPF.
                    graph_instance.update_state(cpf_config, {
                        "messages": result.get("messages", []),
                        "is_authenticated": True,
                        "current_user_cpf": new_cpf,
                        "current_user_name": name,
                        "auth_attempts": 0,
                        "pending_cpf": "",
                        "current_agent": "triage",
                        "routing_target": "",
                    }, as_node="triage")
                    active_thread_id = new_cpf
            except Exception as e:
                logger.error(f"Erro no Thread Hopping: {e}")
                # Fallback: Apenas confia na resposta gerada (Welcome padrão)
                pass
                
        # Extrai a resposta real final do Agente ativo
        ai_messages = [m for m in result.get("messages", []) if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None)]
        if ai_messages:
            last_response = sanitize_internal_tags(ai_messages[-1].content)
        else:
            last_response = "Desculpe, sistema indisponível no momento."
            
        return ChatResponse(
            response=last_response,
            thread_id=active_thread_id,
            is_authenticated=is_now_authenticated,
            current_agent=result.get("current_agent", "triage"),
            cpf=new_cpf,
            name=name
        )
    except asyncio.TimeoutError:
        logger.error(f"Timeout de {INVOKE_TIMEOUT}s atingido no invoke do LangGraph")
        raise HTTPException(status_code=504, detail=f"O sistema demorou mais de {INVOKE_TIMEOUT}s para responder. Tente novamente.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal LangGraph Error: {str(e)}")


@app.post("/api/reset")
async def reset_endpoint():
    """
    Limpa o banco de dados de checkpoints e reconstrói o grafo.
    Útil para testes limpos sem reiniciar o servidor.
    """
    global graph_instance, db_conn
    try:
        # Fecha a conexão atual
        if db_conn:
            db_conn.close()

        # Remove os arquivos do banco
        import tempfile
        db_path = os.path.join(tempfile.gettempdir(), "banco_agil.db")
        for suffix in ["", "-shm", "-wal"]:
            try:
                os.remove(db_path + suffix)
            except FileNotFoundError:
                pass

        # Reconstrói o grafo com banco limpo
        graph_instance, db_conn = build_graph()
        logger.info("Banco de dados resetado com sucesso")
        return {"status": "ok", "message": "Banco de dados limpo e grafo reconstruído."}
    except Exception as e:
        logger.error(f"Erro ao resetar banco: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao resetar: {str(e)}")

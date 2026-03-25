from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import sys
import os

# Adiciona o diretório atual ao path para importação dos agentes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from state import build_graph
from langchain_core.messages import HumanMessage, AIMessage

# Definição dos modelos de entrada e saída
class ChatRequest(BaseModel):
    message: str
    thread_id: str = None  # Enviado pelo frontend (pode ser temporário)
    cpf: str = None        # Opcional, usado se já estiver logado no front

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
graph_instance, db_conn = build_graph(db_path="banco_agil.db")

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
        result = graph_instance.invoke(input_data, config=config)
        
        is_now_authenticated = result.get("is_authenticated", False)
        new_cpf = result.get("current_user_cpf", "")
        name = result.get("current_user_name", "")
        
        # ==============================================================
        # THREAD HOPPING LOGIC (Refatorado do Streamlit direto para a API)
        # ==============================================================
        if not was_authenticated and is_now_authenticated and new_cpf:
            cpf_config = {"configurable": {"thread_id": new_cpf}}
            cpf_state = graph_instance.get_state(cpf_config)
            
            # Se já exisita histórico salvo em disco SQLite no CPF
            if cpf_state and hasattr(cpf_state, "values") and cpf_state.values and len(cpf_state.values.get("messages", [])) > 0:
                welcome_msg = f"Bem-vindo de volta, {name}! 🥳 Recuperei o histórico da nossa última conversa. Como posso ajudar hoje?"
                
                # Injeta a mensagem de retorno na timeline histórica real daquele CPF
                graph_instance.update_state(cpf_config, {"messages": [AIMessage(content=welcome_msg)]})
                
                return ChatResponse(
                    response=welcome_msg,
                    thread_id=new_cpf,
                    is_authenticated=True,
                    current_agent=cpf_state.values.get("current_agent", "triage"),
                    cpf=new_cpf,
                    name=name
                )
            else:
                # É a primeira vez do cliente. "Assumimos" a posse do thread temporal atual colando-o no CPF.
                graph_instance.update_state(cpf_config, result)
                active_thread_id = new_cpf
                
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal LangGraph Error: {str(e)}")

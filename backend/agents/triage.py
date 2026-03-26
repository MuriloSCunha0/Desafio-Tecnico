from langchain_core.messages import SystemMessage, AIMessage
from tools import authenticate_user, end_conversation
from agents.core import (
    get_llm, _get_last_human_content, _detect_routing_target,
    _last_msg_is_tool, _last_msg_is_human, _extract_dob_from_last_human,
    _extract_cpf, _make_tool_call_message, _invoke_with_retry, _trim_messages
)

TRIAGE_CPF_PROMPT = """Você é o Bia, assistente do Banco Ágil.
Cumprimente de forma calorosa e natural — sem ser formal demais. Peça o CPF (só números).
Se quiser encerrar: end_conversation.
Varie o cumprimento, não use sempre o mesmo. Emoji leve, sem exagero. Português do Brasil."""

TRIAGE_DOB_PROMPT = """Você é o Bia, assistente do Banco Ágil.
CPF recebido: {cpf}
Agradeça de forma natural e peça a data de nascimento (DD/MM/AAAA). Não peça o CPF de novo.
Seja breve e amigável, como uma conversa real. Português do Brasil."""

TRIAGE_AUTH_CALL_PROMPT = """Você é o Bia, assistente do Banco Ágil.
Chame authenticate_user com cpf="{cpf}" e date_of_birth="{dob}".
SUCESSO → seja caloroso, chame pelo primeiro nome, pergunte o que deseja de forma descontraída.
FALHA → seja empático e humano, não robótico.
Português do Brasil."""

TRIAGE_RETRY_PROMPT = """Você é o Bia, assistente do Banco Ágil.
CPF: {cpf} | Tentativas restantes: {remaining}
Autenticação falhou. Seja compreensivo e natural — erros acontecem. Peça a data novamente (DD/MM/AAAA).
Se quiser encerrar: end_conversation. Português do Brasil."""

TRIAGE_ROUTING_PROMPT = """Você é o Bia, assistente do Banco Ágil.
Cliente: {name}
Cumprimente pelo primeiro nome de forma descontraída (ex: "E aí, João!" ou "Oi, Maria!").
Pergunte o que deseja. Serviços disponíveis: limites/score 💳, crédito 📈, reavaliação 📊, câmbio 💱.
Seja natural — não liste serviços de forma mecânica. Se quiser encerrar: end_conversation.
Português do Brasil."""

def triage_agent(state: dict) -> dict:
    llm = get_llm()
    is_authenticated = state.get("is_authenticated", False)
    pending_cpf      = state.get("pending_cpf", "")
    auth_attempts    = state.get("auth_attempts", 0)
    user_name        = state.get("current_user_name", "")
    messages         = state.get("messages", [])

    if auth_attempts >= 3:
        block_msg = AIMessage(content=(
            "Sinto muito, mas por questões de segurança não foi possível confirmar "
            "sua identidade após 3 tentativas. Precisamos encerrar este atendimento.\n\n"
            "Você pode nos visitar em qualquer agência do Banco Ágil com um documento "
            "com foto. Tenha um ótimo dia!"
        ))
        return {"messages": [block_msg], "current_agent": "blocked", "routing_target": ""}

    if is_authenticated:
        user_msg    = _get_last_human_content(messages)
        target      = _detect_routing_target(user_msg, messages)
        current     = state.get("current_agent", "triage")

        if not target and current in ("credit", "interview", "forex"):
            return {
                "messages":       [],
                "current_agent":  "triage",
                "routing_target": current,
            }

        if target:
            first_name = user_name.split()[0] if user_name else user_name
            bridge_msgs = {
                "credit":    f"Claro, {first_name}! Vou verificar as informações de crédito agora.",
                "forex":     f"Claro, {first_name}! Vou buscar a cotação atualizada para você.",
                "interview": f"Claro, {first_name}! Vou iniciar a entrevista de reavaliação de crédito.",
            }
            response = AIMessage(content=bridge_msgs[target])
            return {
                "messages":       [response],
                "current_agent":  "triage",
                "routing_target": target,
            }

        system_prompt  = TRIAGE_ROUTING_PROMPT.format(name=user_name)
        llm_with_tools = llm.bind_tools([end_conversation])
        msgs     = [SystemMessage(content=system_prompt)] + _trim_messages(messages)
        response = _invoke_with_retry(llm_with_tools, msgs)
        return {"messages": [response], "current_agent": "triage", "routing_target": ""}

    if _last_msg_is_tool(messages):
        remaining = max(0, 3 - auth_attempts)
        if pending_cpf and auth_attempts > 0:
            system_prompt = TRIAGE_RETRY_PROMPT.format(
                cpf=pending_cpf, remaining=remaining
            )
        else:
            system_prompt = TRIAGE_DOB_PROMPT.format(cpf=pending_cpf or "???")
        llm_with_tools = llm.bind_tools([end_conversation])
        msgs     = [SystemMessage(content=system_prompt)] + _trim_messages(messages)
        response = _invoke_with_retry(llm_with_tools, msgs)
        return {"messages": [response], "current_agent": "triage", "routing_target": ""}

    if pending_cpf and _last_msg_is_human(messages):
        dob = _extract_dob_from_last_human(messages)
        if dob:
            return {
                "messages": [_make_tool_call_message(
                    "authenticate_user", {"cpf": pending_cpf, "date_of_birth": dob}
                )],
                "current_agent": "triage",
                "routing_target": "",
            }

        system_prompt  = TRIAGE_DOB_PROMPT.format(cpf=pending_cpf)
        llm_with_tools = llm.bind_tools([end_conversation])
        msgs     = [SystemMessage(content=system_prompt)] + _trim_messages(messages)
        response = _invoke_with_retry(llm_with_tools, msgs)
        return {"messages": [response], "current_agent": "triage", "routing_target": ""}

    if _last_msg_is_human(messages):
        cpf = _extract_cpf(messages)
        if cpf:
            system_prompt = TRIAGE_DOB_PROMPT.format(cpf=cpf)
            msgs     = [SystemMessage(content=system_prompt)] + _trim_messages(messages)
            response = _invoke_with_retry(llm, msgs)
            return {
                "messages":       [response],
                "current_agent":  "triage",
                "pending_cpf":    cpf,
                "routing_target": "",
            }

        system_prompt  = TRIAGE_CPF_PROMPT
        llm_with_tools = llm.bind_tools([end_conversation])
        msgs     = [SystemMessage(content=system_prompt)] + _trim_messages(messages)
        response = _invoke_with_retry(llm_with_tools, msgs)
        return {"messages": [response], "current_agent": "triage", "routing_target": ""}

    system_prompt  = TRIAGE_CPF_PROMPT
    llm_with_tools = llm.bind_tools([end_conversation])
    msgs     = [SystemMessage(content=system_prompt)] + messages
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response], "current_agent": "triage", "routing_target": ""}

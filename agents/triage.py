from langchain_core.messages import SystemMessage, AIMessage
from tools import authenticate_user, end_conversation
from agents.core import (
    get_llm, _get_last_human_content, _detect_routing_target,
    _last_msg_is_tool, _last_msg_is_human, _extract_dob_from_last_human,
    _extract_cpf, _make_tool_call_message
)

TRIAGE_CPF_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil.

TAREFA: Cumprimente o cliente de forma acolhedora e peça o CPF (somente os 11 números, sem pontos ou traços).

Regras:
- Não peça data de nascimento ainda.
- Se o cliente quiser encerrar, chame end_conversation.
- Não chame nenhuma outra ferramenta.

Tom: respeitoso, acessível. Português do Brasil."""

TRIAGE_DOB_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil.

CPF recebido: {cpf}

TAREFA: Agradeça e peça a data de nascimento no formato DD/MM/AAAA.

Regras:
- Não peça o CPF novamente. Você já tem: {cpf}
- Não invente nem suponha a data.
- NÃO chame nenhuma ferramenta. Apenas responda com texto pedindo a data.

Tom: gentil. Português do Brasil."""

TRIAGE_AUTH_CALL_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil.

AÇÃO IMEDIATA: Chame agora a ferramenta authenticate_user com os dados abaixo.

  cpf = "{cpf}"
  date_of_birth = "{dob}"

Após o resultado:
- SUCESSO → cumprimente o cliente pelo nome e pergunte como pode ajudar.
- FALHA → informe gentilmente que os dados não conferem.

Tom: gentil. Português do Brasil."""

TRIAGE_RETRY_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil.

CPF: {cpf}
Tentativas restantes: {remaining}

A autenticação falhou. Informe gentilmente e peça a data de nascimento novamente (DD/MM/AAAA).
Se o cliente quiser encerrar, chame end_conversation.

Tom: gentil. Português do Brasil."""

TRIAGE_ROUTING_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil.
Cliente: {name}

Cumprimente pelo nome e pergunte como pode ajudar. Serviços disponíveis:
- Consultar score e limite de crédito
- Solicitar aumento de limite de crédito
- Reavaliação de score por entrevista financeira
- Consultar cotação de moedas estrangeiras

Se o cliente quiser encerrar, chame end_conversation.
Tom: acolhedor. Português do Brasil."""

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
        msgs     = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(msgs)
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
        msgs     = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(msgs)
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
        msgs     = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], "current_agent": "triage", "routing_target": ""}

    if _last_msg_is_human(messages):
        cpf = _extract_cpf(messages)
        if cpf:
            system_prompt = TRIAGE_DOB_PROMPT.format(cpf=cpf)
            msgs     = [SystemMessage(content=system_prompt)] + messages
            response = llm.invoke(msgs)
            return {
                "messages":       [response],
                "current_agent":  "triage",
                "pending_cpf":    cpf,
                "routing_target": "",
            }

        system_prompt  = TRIAGE_CPF_PROMPT
        llm_with_tools = llm.bind_tools([end_conversation])
        msgs     = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], "current_agent": "triage", "routing_target": ""}

    system_prompt  = TRIAGE_CPF_PROMPT
    llm_with_tools = llm.bind_tools([end_conversation])
    msgs     = [SystemMessage(content=system_prompt)] + messages
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response], "current_agent": "triage", "routing_target": ""}

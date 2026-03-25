from langchain_core.messages import SystemMessage, AIMessage
from tools import authenticate_user, end_conversation
from agents.core import (
    get_llm, _get_last_human_content, _detect_routing_target,
    _last_msg_is_tool, _last_msg_is_human, _extract_dob_from_last_human,
    _extract_cpf, _make_tool_call_message
)

TRIAGE_CPF_PROMPT = """Você é o Agilito, o assistente virtual parceiro do Banco Ágil.

TAREFA: Recepção inicial. Cumprimente o cliente com entusiasmo e muita clareza, e peça o CPF (avise que pode ser só os números).

Regras:
- Não peça data de nascimento ainda.
- Se o cliente quiser encerrar, chame end_conversation.
- Não chame nenhuma outra ferramenta.

Tom: Caloroso, rápido e super humano. Pode usar emoji (ex: 🏦 ou 👋). Português do Brasil."""

TRIAGE_DOB_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil.

CPF recebido: {cpf}

TAREFA: Agradeça por fornecer o CPF. Agora, para garantir a segurança da conta, peça a data de nascimento no formato DD/MM/AAAA.

Regras:
- Não peça o CPF novamente. Você já tem: {cpf}
- Não invente nem suponha a data.
- NÃO chame nenhuma ferramenta. Apenas responda com texto batendo um papo e pedindo a data.

Tom: Super amigável e focado em segurança 🔒. Português do Brasil."""

TRIAGE_AUTH_CALL_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil.

AÇÃO IMEDIATA: Chame agora a ferramenta authenticate_user com os dados abaixo.

  cpf = "{cpf}"
  date_of_birth = "{dob}"

Após o resultado:
- SUCESSO → comemore a autenticação 🎉, chame o cliente pelo primeiro nome e pergunte animadamente o que ele deseja fazer hoje.
- FALHA → seja muito empático, peça desculpas informando que os dados não bateram.

Tom: Entusiástico e prestativo. Português do Brasil."""

TRIAGE_RETRY_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil.

CPF: {cpf}
Tentativas restantes: {remaining}

A autenticação falhou. Seja compreensivo e diga que é normal errar. Peça gentilmente a data de nascimento de novo (DD/MM/AAAA) avisando sobre as tentativas.
Se o cliente quiser encerrar, chame end_conversation.

Tom: Empático e encorajador. Português do Brasil."""

TRIAGE_ROUTING_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil.
Cliente: {name}

Cumprimente pelo nome (de forma bem pessoal, ex "E aí [nome], como estamos?") e pergunte como pode ajudar.
Serviços disponíveis:
- Ver limites e score 💳
- Pedir mais crédito 📈
- Fazer uma reavaliação (entrevista rápida) 📊
- Checar cotação de moedas 💱

Se o cliente quiser encerrar, chame end_conversation.
Tom: Extrovertido, acolhedor e resolutivo. Português do Brasil."""

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

"""
agents.py — Definição dos agentes do Banco Ágil.

Roteamento robusto para modelos pequenos:
  - Autenticação em 4 fases controladas por Python (authenticate_user só
    fica disponível quando CPF e DOB já foram coletados).
  - Roteamento pós-auth via campo `routing_target` no estado, detectado
    por palavras-chave em Python — sem depender de tags geradas pelo LLM.
  - ROTA: tags via LLM são mantidas apenas como fallback para roteamento
    inter-agentes (interview → credit).
"""

import os
import re
import uuid
from dotenv import load_dotenv
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage
)

load_dotenv()


# ============================================================
# Factory de LLM
# ============================================================

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


# ============================================================
# Utilitários Python — sem depender do LLM
# ============================================================

def _normalize_content(content) -> str:
    """Extrai texto puro de AIMessage.content (compatível com Google Gemini)."""
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
    """Extrai CPF (11 dígitos) da última HumanMessage."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            cleaned = re.sub(r'[.\-\s]', '', str(msg.content))
            match = re.search(r'\d{11}', cleaned)
            if match:
                return match.group()
    return ""


def _extract_dob_from_last_human(messages: list) -> str:
    """Extrai data de nascimento (DD/MM/AAAA) da última HumanMessage."""
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
    """
    Detecta intenção de roteamento por palavras-chave.
    Retorna "credit", "interview", "forex" ou "".

    Para respostas ambíguas como "sim", analisa o contexto do histórico
    de mensagens para inferir a qual oferta o usuário está respondendo.
    """
    msg = user_msg.lower()

    # Intenção de encerramento — deixa o LLM chamar end_conversation
    if any(w in msg for w in ["encerrar", "encerra", "finalizar", "finaliza",
                               "tchau", "até logo", "ate logo", "sair", "fechar"]):
        return ""

    # ── PRIORIDADE 1: Entrevista / Reavaliação ──
    # Deve ser checado ANTES de "crédito" porque "entrevista de crédito" contém ambos.
    if any(w in msg for w in ["entrevista", "reavalia", "melhorar score",
                               "melhorar meu score", "aumentar score",
                               "reavaliação", "reavaliar",
                               "entrevista de crédito", "entrevista financeira"]):
        return "interview"

    # ── PRIORIDADE 2: Crédito (limite, aumento, saldo) ──
    if any(w in msg for w in ["limite", "crédito", "credito", "aumento", "saldo"]):
        return "credit"

    # ── PRIORIDADE 3: Câmbio ──
    if any(w in msg for w in ["câmbio", "cambio", "dólar", "dolar", "euro",
                               "libra", "iene", "moeda", "cotação", "cotacao"]):
        return "forex"

    # ── Score / Pontuação: interview se reavaliar, credit para consulta ──
    if any(w in msg for w in ["score", "pontuação", "pontuacao"]):
        return "credit"   # mostra score atual; credit agent oferece entrevista se necessário

    # Detecção contextual para respostas afirmativas curtas ("sim", "ok", "quero")
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
    """Cria uma AIMessage com tool_call forçada, sem passar pelo LLM."""
    return AIMessage(
        content="",
        tool_calls=[{
            "id": str(uuid.uuid4()),
            "name": tool_name,
            "args": args,
        }],
    )


def _detect_currency(messages: list) -> str:
    """Detecta código de moeda a partir das mensagens do usuário ou, por contexto,
    da última AIMessage quando o usuário responde com afirmativa curta."""
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
            # Resposta afirmativa curta — verificar contexto da última AIMessage
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


def _extract_amount(text: str) -> float:
    """Extrai valor monetário de um texto."""
    cleaned = re.sub(r'[R$\s]', '', text)
    match = re.search(r'\d[\d.,]*', cleaned)
    if match:
        try:
            return float(match.group().replace('.', '').replace(',', '.'))
        except ValueError:
            pass
    return 0.0


def _has_tool_result(messages: list, substring: str) -> bool:
    """Verifica se alguma ToolMessage contém determinada substring."""
    return any(
        isinstance(m, ToolMessage) and substring in str(m.content)
        for m in messages
    )


def _has_currency_result(messages: list, currency: str) -> bool:
    """Verifica se já existe cotação para a moeda nas ToolMessages."""
    return any(
        isinstance(m, ToolMessage) and f"1 {currency}" in str(m.content)
        for m in messages
    )


def _parse_employment(t: str) -> str:
    """Extrai tipo de emprego de texto, retorna '' se não encontrado."""
    if "desempregado" in t or "sem emprego" in t or "sem trabalho" in t:
        return "desempregado"
    if any(w in t for w in ["carteira assinada", "clt", "formal", "registrado", "empregado"]):
        return "formal"
    if any(w in t for w in ["autônomo", "autonomo", "freelancer", "bico",
                             "conta própria", "conta propria", "por conta"]):
        return "autônomo"
    return ""


def _parse_debts(t: str, short_answer: bool = False) -> str:
    """Extrai resposta de dívidas de texto, retorna '' se não encontrado.
    short_answer=True permite interpretar 'sim'/'não' isolados (contexto confirmado).
    """
    if re.search(r'n[aã]o\s+(?:tenho|possuo)\s+d[ií]vidas?', t) or \
       re.search(r'sem\s+d[ií]vidas?', t) or \
       re.search(r'nenhuma\s+d[ií]vidas?', t):
        return "não"
    if re.search(r'(?:tenho|possuo)\s+d[ií]vidas?', t):
        return "sim"
    # Resposta curta só é interpretada quando vem de extração contextual
    if short_answer:
        if re.search(r'^n[aã]o\.?$', t.strip()):
            return "não"
        if re.search(r'^sim\.?$', t.strip()):
            return "sim"
    return ""


def _parse_number(t: str) -> float:
    """Extrai primeiro número válido do texto, desconsiderando R$."""
    cleaned = re.sub(r'[R$]', '', t)
    m = re.search(r'(\d[\d.,]*)', cleaned)
    if m:
        try:
            return float(m.group(1).replace('.', '').replace(',', '.'))
        except ValueError:
            pass
    return -1.0


def _parse_interview_fields(text: str) -> dict:
    """
    Extrai os 5 campos da entrevista de crédito a partir de texto livre.
    Retorna dict com os campos que conseguiu identificar.
    """
    t = text.lower()
    result = {}

    # 1. Renda mensal — busca com palavras-chave
    m = re.search(
        r'(?:ganho|renda|sal[aá]rio|recebo|fa[çc]o|recebi|minha renda)[^\d]*(?:r\$\s*)?(\d[\d.,]*)', t
    )
    if not m:
        m = re.search(r'(\d[\d.,]+)\s*(?:por\s*m[eê]s|mensais|reais\s*/?\s*m[eê]s)', t)
    if m:
        try:
            result["monthly_income"] = float(m.group(1).replace('.', '').replace(',', '.'))
        except ValueError:
            pass

    # 2. Tipo de emprego
    emp = _parse_employment(t)
    if emp:
        result["employment_type"] = emp

    # 3. Despesas mensais — busca com palavras-chave
    m = re.search(
        r'(?:gastos?\s*(?:fixos?)?|despesas?\s*(?:fixas?)?|custo)[^\d]*(?:r\$\s*)?(\d[\d.,]*)', t
    )
    if not m:
        m = re.search(r'(?:tenho|s[ãa]o)\s+(?:r\$\s*)?(\d[\d.,]+)\s+(?:de\s+)?(?:despesas?|gastos?)', t)
    if m:
        try:
            result["monthly_expenses"] = float(m.group(1).replace('.', '').replace(',', '.'))
        except ValueError:
            pass

    # 4. Dependentes
    if any(p in t for p in ["sem dependentes", "não tenho dependentes", "nao tenho dependentes",
                              "0 dependentes", "nenhum dependente", "nenhuma dependente", "nenhum"]):
        result["dependents"] = 0
    else:
        m = re.search(r'(\d+)\s+dependentes?', t)
        if m:
            result["dependents"] = int(m.group(1))

    # 5. Dívidas
    debts = _parse_debts(t)
    if debts:
        result["has_debts"] = debts

    return result


def _parse_interview_fields_context(messages: list) -> dict:
    """
    Extrai os 5 campos da entrevista analisando pares (pergunta AI → resposta usuário).

    Estratégia correta: escaneia em ordem CRONOLÓGICA e cada HumanMessage pode ser
    usada para apenas UM campo. Isso evita que mensagens de confirmação da AI (que
    mencionam o campo anterior + a próxima pergunta) causem extração errada.
    """
    # 1. Extração por palavras-chave no texto combinado (rápida, cobre frases completas)
    combined = " ".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    result = _parse_interview_fields(combined)

    # Gatilhos por campo — do mais específico ao mais genérico
    # A ordem importa: ao escanear, o gatilho mais específico evita falsos positivos
    field_configs = [
        ("monthly_income",   ["renda mensal", "renda", "salário", "salario", "quanto ganha"]),
        ("employment_type",  ["tipo de emprego", "carteira assinada", "autônomo", "formal"]),
        ("monthly_expenses", ["despesas fixas", "despesas mensais", "despesas", "gastos", "quanto gasta"]),
        ("dependents",       ["quantos dependentes", "número de dependentes", "dependentes"]),
        ("has_debts",        ["dívidas ativas", "dívidas", "dividas ativas", "dividas"]),
    ]

    # Montar lista de pares (índice_AI, índice_Human) em ordem cronológica
    pairs = []
    for i in range(len(messages) - 1):
        if isinstance(messages[i], AIMessage) and isinstance(messages[i + 1], HumanMessage):
            pairs.append((i, i + 1))

    # Conjunto de índices de HumanMessage já consumidos (1 campo por resposta)
    used_human_indices: set = set()

    # 2. Para cada campo ainda não extraído, busca o par mais ANTIGO (cronológico) que o cobre
    for field, triggers in field_configs:
        if field in result:
            continue

        for ai_idx, human_idx in pairs:
            if human_idx in used_human_indices:
                continue

            ai_text    = _normalize_content(messages[ai_idx].content).lower()
            human_text = str(messages[human_idx].content).lower().strip()

            if not any(tr in ai_text for tr in triggers):
                continue  # este AI message não perguntou sobre este campo

            extracted = False

            if field == "monthly_income":
                v = _parse_number(human_text)
                if v > 0:
                    result["monthly_income"] = v
                    extracted = True

            elif field == "employment_type":
                emp = _parse_employment(human_text)
                if emp:
                    result["employment_type"] = emp
                    extracted = True

            elif field == "monthly_expenses":
                v = _parse_number(human_text)
                if v >= 0:
                    result["monthly_expenses"] = v
                    extracted = True

            elif field == "dependents":
                if any(p in human_text for p in ["nenhum", "sem", "não tenho",
                                                   "nao tenho", "zero"]):
                    result["dependents"] = 0
                    extracted = True
                else:
                    # Aceita apenas dígito(s) sozinhos ou "N dependentes"
                    m = re.search(r'^(\d+)$', human_text.strip())
                    if not m:
                        m = re.search(r'(\d+)\s*dependentes?', human_text)
                    if m:
                        result["dependents"] = int(m.group(1))
                        extracted = True

            elif field == "has_debts":
                debts = _parse_debts(human_text, short_answer=True)
                if debts:
                    result["has_debts"] = debts
                    extracted = True

            if extracted:
                used_human_indices.add(human_idx)
                break  # encontrou o par mais recente para este campo; próximo campo

    return result


# ============================================================
# Prompts — Agente de Triagem (um por fase)
# ============================================================

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


# ============================================================
# Prompts — Agentes especializados
# ============================================================

CREDIT_SYSTEM_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil — especialista em crédito.
{name_line}

Ferramentas:
- check_limit: consulta o limite e score atual
- request_limit_increase: solicita aumento de limite

Fluxo:
1. Use check_limit para ver o limite e score atuais, se ainda não fez.
2. Informe os dados ao cliente (limite e score).
3. Se o cliente quiser AUMENTAR o limite: pergunte o valor desejado, depois chame request_limit_increase.
   - APROVADO: parabenize e informe o novo limite.
   - REJEITADO: explique o motivo, ofereça entrevista de reavaliação de score e aguarde resposta.
4. Se o cliente quiser apenas CONSULTAR (score, limite, saldo): informe os dados e pergunte se deseja mais algo.
   NÃO ofereça aumento de limite automaticamente — espere o cliente pedir.

Se o cliente quiser encerrar, chame end_conversation.
Tom: respeitoso, didático. Português do Brasil."""


INTERVIEW_COLLECT_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil — entrevista de reavaliação de crédito.
{name_line}
CPF do cliente: {cpf}

TAREFA: Coletar as 5 informações abaixo e, quando tiver TODAS, chamar calculate_and_update_score.

Informações necessárias:
1. Renda mensal (em reais)
2. Tipo de emprego: formal (CLT/carteira assinada), autônomo ou desempregado
3. Despesas fixas mensais (em reais)
4. Número de dependentes (0, 1, 2 ou 3+)
5. Possui dívidas ativas? (sim ou não)

REGRAS OBRIGATÓRIAS:
- Pergunte APENAS o que ainda falta. Uma pergunta por vez.
- Aceite respostas informais: "ganho dois mil"→2000, "CLT"→formal, "faço bico"→autônomo.
- QUANDO TIVER TODOS OS 5 DADOS: chame calculate_and_update_score IMEDIATAMENTE.
  NÃO diga "vou processar", "aguarde", "calculando" nem descreva o resultado em texto.
  Você NÃO consegue calcular score — somente a ferramenta calculate_and_update_score faz isso.
  NUNCA invente ou estime valores de score ou limite. Chame a ferramenta e aguarde o resultado.
- NÃO adicione ROTA:.
- Se o cliente quiser encerrar, chame end_conversation.

Tom: paciente, encorajador. Português do Brasil."""


INTERVIEW_RESULT_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil.
{name_line}

A reavaliação de crédito foi concluída. Comunique o resultado de forma clara e positiva.
Informe o novo score e o novo limite de crédito disponível.

Tom: entusiasmado, encorajador. Português do Brasil."""


FOREX_SYSTEM_PROMPT = """Você é o Agilito, assistente virtual do Banco Ágil — especialista em câmbio.
{name_line}

Use get_currency_rate para cotações. Traduza: dólar→USD, euro→EUR, libra→GBP, iene→JPY.
Após a cotação, pergunte se precisa de mais algo.
Se o cliente quiser converter um valor, calcule e explique.
Para encerrar, use end_conversation.

Tom: acessível, claro. Português do Brasil."""


# ============================================================
# Importação das ferramentas
# ============================================================

from tools import (
    authenticate_user,
    check_limit,
    request_limit_increase,
    calculate_and_update_score,
    get_currency_rate,
    end_conversation,
)


# ============================================================
# Agente de Triagem — máquina de 4 fases
# ============================================================

def triage_agent(state: dict) -> dict:
    """
    Fases de autenticação:
      A. Sem CPF         → pede CPF (só end_conversation disponível)
      B. Tem CPF, sem DOB → pede DOB (só end_conversation disponível)
      C. Tem CPF + DOB   → chama authenticate_user com valores injetados no prompt
      D. Autenticado     → detecta intenção via Python, define routing_target no estado
    """
    llm = get_llm()

    is_authenticated = state.get("is_authenticated", False)
    pending_cpf      = state.get("pending_cpf", "")
    auth_attempts    = state.get("auth_attempts", 0)
    user_name        = state.get("current_user_name", "")
    messages         = state.get("messages", [])

    # Bloqueio após 3 tentativas
    if auth_attempts >= 3:
        block_msg = AIMessage(content=(
            "Sinto muito, mas por questões de segurança não foi possível confirmar "
            "sua identidade após 3 tentativas. Precisamos encerrar este atendimento.\n\n"
            "Você pode nos visitar em qualquer agência do Banco Ágil com um documento "
            "com foto. Tenha um ótimo dia!"
        ))
        return {"messages": [block_msg], "current_agent": "blocked", "routing_target": ""}

    # ── Fase D: Autenticado — roteamento via Python ─────────────────────────
    if is_authenticated:
        user_msg    = _get_last_human_content(messages)
        target      = _detect_routing_target(user_msg, messages)
        current     = state.get("current_agent", "triage")

        # Se não detectou intenção nova mas está no meio de conversa com
        # agente especializado → continuar com ele (ex.: dados da entrevista)
        if not target and current in ("credit", "interview", "forex"):
            return {
                "messages":       [],          # sem mensagem de transição
                "current_agent":  "triage",
                "routing_target": current,
            }

        if target:
            # Intenção detectada: gera mensagem de transição e define routing_target
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

        # Intenção não detectada — LLM pergunta o que o cliente quer
        system_prompt  = TRIAGE_ROUTING_PROMPT.format(name=user_name)
        llm_with_tools = llm.bind_tools([end_conversation])
        msgs     = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], "current_agent": "triage", "routing_target": ""}

    # ── Após execução de ferramenta: gerar resposta pós-tool ────────────────
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

    # ── Fase C: Temos CPF e usuário forneceu data ────────────────────────────
    if pending_cpf and _last_msg_is_human(messages):
        dob = _extract_dob_from_last_human(messages)
        if dob:
            system_prompt  = TRIAGE_AUTH_CALL_PROMPT.format(cpf=pending_cpf, dob=dob)
            llm_with_tools = llm.bind_tools([authenticate_user, end_conversation])
            msgs     = [SystemMessage(content=system_prompt)] + messages
            response = llm_with_tools.invoke(msgs)
            return {"messages": [response], "current_agent": "triage", "routing_target": ""}

        # Usuário respondeu mas sem data — pedir DOB novamente
        system_prompt  = TRIAGE_DOB_PROMPT.format(cpf=pending_cpf)
        llm_with_tools = llm.bind_tools([end_conversation])
        msgs     = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], "current_agent": "triage", "routing_target": ""}

    # ── Fase A/B: Sem pending_cpf ────────────────────────────────────────────
    if _last_msg_is_human(messages):
        cpf = _extract_cpf(messages)
        if cpf:
            # CPF detectado — pedir DOB sem ferramentas disponíveis
            # (evita que modelos pequenos chamem end_conversation ao receber número)
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

    # ── Fallback ─────────────────────────────────────────────────────────────
    system_prompt  = TRIAGE_CPF_PROMPT
    llm_with_tools = llm.bind_tools([end_conversation])
    msgs     = [SystemMessage(content=system_prompt)] + messages
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response], "current_agent": "triage", "routing_target": ""}


# ============================================================
# Agente de Crédito
# ============================================================

def credit_agent(state: dict) -> dict:
    messages = state["messages"]
    cpf      = state.get("current_user_cpf", "")

    check_done   = _has_tool_result(messages, "Dados de crédito")
    increase_done = (
        _has_tool_result(messages, "APROVADO") or
        _has_tool_result(messages, "REJEITADO")
    )

    # ── Python-driven tool injection ───────────────────────────────────────
    if not increase_done:
        last_human   = _get_last_human_content(messages)
        amount       = _extract_amount(last_human)
        increase_kws = ["aumentar", "aumento", "quero", "gostaria", "preciso",
                        "solicitar", "limite de", "novo limite"]

        # Usuário informou valor específico → solicitar aumento direto
        if amount > 0 and any(w in last_human.lower() for w in increase_kws):
            return {
                "messages": [_make_tool_call_message(
                    "request_limit_increase", {"cpf": cpf, "requested_value": amount}
                )],
                "current_agent": "credit",
                "routing_target": "",
            }

        # Sem valor ainda, e limite não consultado → consultar primeiro
        if not check_done:
            return {
                "messages": [_make_tool_call_message("check_limit", {"cpf": cpf})],
                "current_agent": "credit",
                "routing_target": "",
            }

    # ── LLM lida com resultados das tools e perguntas de acompanhamento ───
    llm = get_llm()
    llm_with_tools = llm.bind_tools([check_limit, request_limit_increase, end_conversation])

    user_name     = state.get("current_user_name", "")
    name_line     = f"Cliente: {user_name}" if user_name else ""
    system_prompt = CREDIT_SYSTEM_PROMPT.format(name_line=name_line)

    msgs     = [SystemMessage(content=system_prompt)] + messages
    response = llm_with_tools.invoke(msgs)

    return {"messages": [response], "current_agent": "credit", "routing_target": ""}


# ============================================================
# Agente de Entrevista
# ============================================================

def interview_agent(state: dict) -> dict:
    messages  = state["messages"]
    cpf       = state.get("current_user_cpf", "")
    user_name = state.get("current_user_name", "")

    name_line = f"Cliente: {user_name}" if user_name else ""

    # ── Score já calculado — LLM comunica resultado, Python roteia para crédito
    if _has_tool_result(messages, "REAVALIAÇÃO CONCLUÍDA"):
        llm = get_llm()
        system_prompt = INTERVIEW_RESULT_PROMPT.format(name_line=name_line)
        msgs     = [SystemMessage(content=system_prompt)] + messages
        response = llm.bind_tools([end_conversation]).invoke(msgs)
        return {
            "messages":       [response],
            "current_agent":  "interview",
            "routing_target": "credit",   # Python roteia, sem depender de LLM
        }

    # ── Python injection: extração contextual (par pergunta→resposta) ───────
    fields   = _parse_interview_fields_context(messages)
    required = ["monthly_income", "employment_type", "monthly_expenses",
                "dependents", "has_debts"]

    if all(k in fields for k in required):
        return {
            "messages": [_make_tool_call_message(
                "calculate_and_update_score",
                {
                    "cpf":               cpf,
                    "monthly_income":    fields["monthly_income"],
                    "employment_type":   fields["employment_type"],
                    "monthly_expenses":  fields["monthly_expenses"],
                    "dependents":        fields["dependents"],
                    "has_debts":         fields["has_debts"],
                },
            )],
            "current_agent": "interview",
            "routing_target": "",
        }

    # ── LLM coleta os dados que faltam (prompt sem menção a ROTA:) ────────
    llm = get_llm()
    system_prompt = INTERVIEW_COLLECT_PROMPT.format(name_line=name_line, cpf=cpf)
    msgs     = [SystemMessage(content=system_prompt)] + messages
    # calculate_and_update_score incluído para que, se o LLM detectar que todos os
    # campos foram coletados, ele use o nome de tool correto em vez de improvisar.
    response = llm.bind_tools([calculate_and_update_score, end_conversation]).invoke(msgs)

    return {"messages": [response], "current_agent": "interview", "routing_target": ""}


# ============================================================
# Agente de Câmbio
# ============================================================

def forex_agent(state: dict) -> dict:
    messages = state["messages"]

    # ── Python-driven tool injection ──────────────────────────────────────
    # Buscar cotação se moeda detectada e ainda não há resultado para ela
    currency = _detect_currency(messages)
    if currency and not _has_currency_result(messages, currency):
        return {
            "messages": [_make_tool_call_message(
                "get_currency_rate", {"currency": currency}
            )],
            "current_agent": "forex",
            "routing_target": "",
        }

    # ── LLM responde com base no resultado da tool ou faz perguntas ───────
    llm = get_llm()
    llm_with_tools = llm.bind_tools([get_currency_rate, end_conversation])

    user_name     = state.get("current_user_name", "")
    name_line     = f"Cliente: {user_name}" if user_name else ""
    system_prompt = FOREX_SYSTEM_PROMPT.format(name_line=name_line)

    msgs     = [SystemMessage(content=system_prompt)] + messages
    response = llm_with_tools.invoke(msgs)

    return {"messages": [response], "current_agent": "forex", "routing_target": ""}

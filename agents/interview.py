import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools import calculate_and_update_score, end_conversation
from agents.core import get_llm, _normalize_content, _has_tool_result, _make_tool_call_message

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

def _parse_employment(t: str) -> str:
    if "desempregado" in t or "sem emprego" in t or "sem trabalho" in t:
        return "desempregado"
    if any(w in t for w in ["carteira assinada", "clt", "formal", "registrado", "empregado"]):
        return "formal"
    if any(w in t for w in ["autônomo", "autonomo", "freelancer", "bico",
                             "conta própria", "conta propria", "por conta"]):
        return "autônomo"
    return ""

def _parse_debts(t: str, short_answer: bool = False) -> str:
    if re.search(r'n[aã]o\s+(?:tenho|possuo)\s+d[ií]vidas?', t) or \
       re.search(r'sem\s+d[ií]vidas?', t) or \
       re.search(r'nenhuma\s+d[ií]vidas?', t):
        return "não"
    if re.search(r'(?:tenho|possuo)\s+d[ií]vidas?', t):
        return "sim"
    if short_answer:
        if re.search(r'^n[aã]o\.?$', t.strip()):
            return "não"
        if re.search(r'^sim\.?$', t.strip()):
            return "sim"
    return ""

def _parse_number(t: str) -> float:
    cleaned = re.sub(r'[R$]', '', t)
    m = re.search(r'(\d[\d.,]*)', cleaned)
    if m:
        try:
            return float(m.group(1).replace('.', '').replace(',', '.'))
        except ValueError:
            pass
    return -1.0

def _parse_interview_fields(text: str) -> dict:
    t = text.lower()
    result = {}

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

    emp = _parse_employment(t)
    if emp:
        result["employment_type"] = emp

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

    if any(p in t for p in ["sem dependentes", "não tenho dependentes", "nao tenho dependentes",
                              "0 dependentes", "nenhum dependente", "nenhuma dependente", "nenhum"]):
        result["dependents"] = 0
    else:
        m = re.search(r'(\d+)\s+dependentes?', t)
        if m:
            result["dependents"] = int(m.group(1))

    debts = _parse_debts(t)
    if debts:
        result["has_debts"] = debts

    return result

def _parse_interview_fields_context(messages: list) -> dict:
    combined = " ".join(str(m.content) for m in messages if isinstance(m, HumanMessage))
    result = _parse_interview_fields(combined)

    field_configs = [
        ("monthly_income",   ["renda mensal", "renda", "salário", "salario", "quanto ganha"]),
        ("employment_type",  ["tipo de emprego", "carteira assinada", "autônomo", "formal"]),
        ("monthly_expenses", ["despesas fixas", "despesas mensais", "despesas", "gastos", "quanto gasta"]),
        ("dependents",       ["quantos dependentes", "número de dependentes", "dependentes"]),
        ("has_debts",        ["dívidas ativas", "dívidas", "dividas ativas", "dividas"]),
    ]

    pairs = []
    for i in range(len(messages) - 1):
        if isinstance(messages[i], AIMessage) and isinstance(messages[i + 1], HumanMessage):
            pairs.append((i, i + 1))

    used_human_indices: set = set()

    for field, triggers in field_configs:
        if field in result:
            continue

        for ai_idx, human_idx in pairs:
            if human_idx in used_human_indices:
                continue

            ai_text    = _normalize_content(messages[ai_idx].content).lower()
            human_text = str(messages[human_idx].content).lower().strip()

            if not any(tr in ai_text for tr in triggers):
                continue

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
                break

    return result

def interview_agent(state: dict) -> dict:
    messages  = state["messages"]
    cpf       = state.get("current_user_cpf", "")
    user_name = state.get("current_user_name", "")

    name_line = f"Cliente: {user_name}" if user_name else ""

    if _has_tool_result(messages, "REAVALIAÇÃO CONCLUÍDA"):
        llm = get_llm()
        system_prompt = INTERVIEW_RESULT_PROMPT.format(name_line=name_line)
        msgs     = [SystemMessage(content=system_prompt)] + messages
        response = llm.bind_tools([end_conversation]).invoke(msgs)
        return {
            "messages":       [response],
            "current_agent":  "interview",
            "routing_target": "credit",
        }

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

    llm = get_llm()
    system_prompt = INTERVIEW_COLLECT_PROMPT.format(name_line=name_line, cpf=cpf)
    msgs     = [SystemMessage(content=system_prompt)] + messages
    response = llm.bind_tools([calculate_and_update_score, end_conversation]).invoke(msgs)

    return {"messages": [response], "current_agent": "interview", "routing_target": ""}

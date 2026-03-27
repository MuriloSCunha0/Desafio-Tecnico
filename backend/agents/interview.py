import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from tools import calculate_and_update_score, end_conversation
from agents.core import get_llm, _normalize_content, _make_tool_call_message, _invoke_with_retry, _trim_messages, _strip_llm_artifacts


def _has_unprocessed_reaval(messages: list) -> bool:
    """Returns True only if the REAVALIAÇÃO CONCLUÍDA tool result hasn't been
    addressed yet in this turn (i.e. no HumanMessage came after the last one)."""
    last_reaval_idx = None
    for i, m in enumerate(messages):
        if isinstance(m, ToolMessage) and "REAVALIAÇÃO CONCLUÍDA" in str(m.content):
            last_reaval_idx = i
    if last_reaval_idx is None:
        return False
    # If any HumanMessage appears after the last reaval result, it was already handled
    for m in messages[last_reaval_idx + 1:]:
        if isinstance(m, HumanMessage):
            return False
    return True

INTERVIEW_COLLECT_PROMPT = """Você é o Bia, consultor financeiro do Banco Ágil.
{name_line}
CPF: {cpf}

Campos já coletados: {already_collected}
Faltam coletar: {missing_fields}

Pergunte sobre ATÉ DOIS campos por mensagem, agrupando os que fazem sentido juntos:
- renda mensal + tipo de emprego (andam juntos — falam do trabalho)
- despesas fixas + dependentes (andam juntos — falam dos gastos)
- dívidas ativas (pergunta sozinha — é sim/não)

Se faltar apenas um campo, pergunte só sobre ele.
Seja natural e conciso — uma frase só por pergunta, como numa conversa.
Aceite respostas informais: "ganho dois mil"→2000, "CLT"→formal, "faço bico"→autônomo, "nenhum"→0 dependentes.

REGRAS CRÍTICAS:
- NUNCA assuma ou invente valores para campos que o cliente ainda não informou.
- NUNCA inclua JSON ou dados estruturados na sua resposta de texto.
- Chame calculate_and_update_score SOMENTE quando TODOS os 5 campos estiverem em "Campos já coletados".
- Se quiser encerrar: end_conversation.

Tom: Leve, empático, curioso. Português do Brasil."""

INTERVIEW_RESULT_PROMPT = """Você é o Bia, consultor financeiro do Banco Ágil.
{name_line}

A reavaliação foi concluída. Compartilhe o resultado de forma genuína e calorosa.
- Score melhorou: comemore de verdade, mostre empolgação real 🎉 — mencione o novo score e novo limite.
- Score igual ou caiu: seja honesto e encorajador, dê uma dica prática de educação financeira.

Seja humano, não robótico. Português do Brasil."""

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

    required = ["monthly_income", "employment_type", "monthly_expenses",
                "dependents", "has_debts"]

    FIELD_LABELS = {
        "monthly_income":   "renda mensal",
        "employment_type":  "tipo de emprego",
        "monthly_expenses": "despesas fixas",
        "dependents":       "dependentes",
        "has_debts":        "dívidas ativas",
    }

    if _has_unprocessed_reaval(messages):
        llm = get_llm()
        system_prompt = INTERVIEW_RESULT_PROMPT.format(name_line=name_line)
        msgs     = [SystemMessage(content=system_prompt)] + _trim_messages(messages)
        response = _invoke_with_retry(llm, msgs)  # No tools — just present the result
        return {
            "messages":       [response],
            "current_agent":  "credit",   # next turn goes to credit
            "routing_target": "",          # END this turn so user sees the result
        }

    fields = _parse_interview_fields_context(messages)

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

    # Resumo do que já foi coletado e o que ainda falta — evita depender do histórico longo
    collected_str = ", ".join(
        f"{FIELD_LABELS[k]}={fields[k]}" for k in required if k in fields
    ) or "nenhum ainda"
    missing_str = ", ".join(
        FIELD_LABELS[k] for k in required if k not in fields
    )

    llm = get_llm()
    system_prompt = INTERVIEW_COLLECT_PROMPT.format(
        name_line=name_line,
        cpf=cpf,
        already_collected=collected_str,
        missing_fields=missing_str,
    )
    msgs     = [SystemMessage(content=system_prompt)] + _trim_messages(messages)
    response = _invoke_with_retry(llm.bind_tools([calculate_and_update_score, end_conversation]), msgs)

    # Groq às vezes vaza JSON / <function=...> no texto — limpa sem usar os valores assumidos.
    # A coleta continua normalmente; o caminho determinístico chama a tool quando tiver tudo.
    cleaned = _strip_llm_artifacts(response.content)
    if cleaned != (response.content or ""):
        response = AIMessage(content=cleaned, tool_calls=[], additional_kwargs={})

    return {"messages": [response], "current_agent": "interview", "routing_target": ""}

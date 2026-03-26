"""
test_full_flow.py — Fluxo completo de atendimento do Banco Ágil.

Cenário único para Carlos Eduardo Lima cobrindo:
  1. Autenticação (CPF + data juntos)
  2. Verificação de limite e score
  3. Aumento de limite → APROVADO
  4. Aumento de limite → REJEITADO
  5. Entrevista de reavaliação de crédito
  6. Consulta de câmbio (USD e EUR)

Uso:
    cd backend && python ../test_full_flow.py
"""

import os
import sys
import uuid

# Garante que o backend está no path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.insert(0, BACKEND_DIR)

from langchain_core.messages import AIMessage, HumanMessage
from state import build_graph

# ============================================================
# Cliente: Carlos Eduardo Lima
#   CPF:    45678901234
#   DOB:    28/02/1978
#   Score:  535 → limite máximo permitido = R$ 5.000
#   Limite atual no CSV: R$ 8.000 (acima do máximo do score)
# ============================================================

SCENARIO = {
    "name": "FLUXO COMPLETO",
    "steps": [
        # ── Autenticação ──────────────────────────────────────────
        {
            "label": "AUTH",
            "input": "45678901234 28/02/1978",
            "note":  "CPF e data de nascimento juntos (novo fluxo unificado)",
        },
        # ── Consulta de limite ───────────────────────────────────
        {
            "label": "CHECK_LIMIT",
            "input": "Quero ver meu limite e meu score",
            "note":  "Deve chamar check_limit e retornar dados reais",
        },
        # ── Aumento APROVADO ─────────────────────────────────────
        {
            "label": "AUMENTO_APROVADO",
            "input": "Quero aumentar meu limite para 4500 reais",
            "note":  "4500 ≤ 5000 (max para score 535) → deve ser APROVADO",
        },
        # ── Aumento REJEITADO ────────────────────────────────────
        {
            "label": "AUMENTO_REJEITADO",
            "input": "Quero agora um limite de 20000 reais",
            "note":  "20000 > 5000 → deve ser REJEITADO",
        },
        # ── Aceitar entrevista ───────────────────────────────────
        {
            "label": "ACEITAR_ENTREVISTA",
            "input": "Quero fazer a entrevista de reavaliação",
            "note":  "Deve rotear para o agente de entrevista",
        },
        # ── Entrevista: renda + emprego ──────────────────────────
        {
            "label": "ENTREVISTA_1",
            "input": "Ganho 6000 por mês, sou CLT",
            "note":  "Captura monthly_income=6000 e employment_type=formal",
        },
        # ── Entrevista: despesas + dependentes ───────────────────
        {
            "label": "ENTREVISTA_2",
            "input": "Gasto 1500 fixo por mês e tenho 1 dependente",
            "note":  "Captura monthly_expenses=1500 e dependents=1",
        },
        # ── Entrevista: dívidas ──────────────────────────────────
        {
            "label": "ENTREVISTA_3",
            "input": "Não tenho dívidas ativas",
            "note":  "Captura has_debts=não → deve chamar calculate_and_update_score",
        },
        # ── Câmbio: dólar ────────────────────────────────────────
        {
            "label": "CAMBIO_USD",
            "input": "Quero ver a cotação do dólar",
            "note":  "Deve chamar get_currency_rate(USD)",
        },
        # ── Câmbio: euro + conversão ─────────────────────────────
        {
            "label": "CAMBIO_EUR",
            "input": "E o euro? Quanto ficaria 300 euros em reais?",
            "note":  "Deve chamar get_currency_rate(EUR) e converter",
        },
        # ── Encerramento ─────────────────────────────────────────
        {
            "label": "ENCERRAR",
            "input": "Pode encerrar, obrigado!",
            "note":  "Deve chamar end_conversation",
        },
    ],
}


def normalize_content(content) -> str:
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content) if content else ""


def print_separator(char="─", width=70):
    print(char * width)


def run():
    graph, _conn = build_graph(":memory:")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    local_state = {
        "is_authenticated":  False,
        "auth_attempts":     0,
        "current_user_cpf":  "",
        "current_user_name": "",
        "pending_cpf":       "",
        "routing_target":    "",
        "interview_context": {},
        "current_agent":     "triage",
    }

    prev_msg_count = 0
    tools_sequence = []

    print()
    print_separator("═")
    print(f"  BANCO ÁGIL — {SCENARIO['name']}")
    print_separator("═")
    print()

    for step in SCENARIO["steps"]:
        label = step["label"]
        user_input = step["input"]
        note = step["note"]

        print_separator()
        print(f"  [{label}]  {note}")
        print_separator()
        print(f"  👤  {user_input}")
        print()

        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_input)], **local_state},
                config=config,
            )

            # Atualizar estado local
            for key in ["is_authenticated", "auth_attempts", "current_user_cpf",
                        "current_user_name", "pending_cpf", "routing_target", "current_agent"]:
                if key in result:
                    local_state[key] = result[key]
            if "interview_context" in result:
                local_state["interview_context"] = result["interview_context"]

            all_messages = result.get("messages", [])
            new_messages  = all_messages[prev_msg_count:]
            prev_msg_count = len(all_messages)

            # Ferramentas chamadas neste turno
            turn_tools = []
            for msg in new_messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        turn_tools.append(tc["name"])
                        tools_sequence.append(tc["name"])

            if turn_tools:
                print(f"  🔧  Ferramentas: {' → '.join(turn_tools)}")
                print()

            # Resposta final visível
            ai_msgs = [
                m for m in new_messages
                if isinstance(m, AIMessage) and m.content and not m.tool_calls
            ]
            if ai_msgs:
                text = normalize_content(ai_msgs[-1].content)
                print(f"  🤖  {text}")
            else:
                print("  🤖  (processando...)")

        except Exception as e:
            print(f"  ❌  ERRO: {e}")

        # Estado resumido após o turno
        print()
        agent = local_state.get("current_agent", "?")
        authed = "✅" if local_state.get("is_authenticated") else "❌"
        print(f"  Estado → agente={agent}  autenticado={authed}")
        print()

    # ── Resumo final ────────────────────────────────────────────
    print_separator("═")
    print("  RESUMO FINAL")
    print_separator("═")
    print(f"  Cliente:      {local_state.get('current_user_name') or '—'}")
    print(f"  CPF:          {local_state.get('current_user_cpf') or '—'}")
    print(f"  Autenticado:  {local_state.get('is_authenticated')}")
    print(f"  Agente final: {local_state.get('current_agent')}")
    print()
    print(f"  Sequência de ferramentas chamadas:")
    print(f"  {' → '.join(tools_sequence) or '(nenhuma)'}")
    print_separator("═")
    print()


if __name__ == "__main__":
    run()

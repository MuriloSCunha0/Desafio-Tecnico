"""
test_flows.py — Testes de fluxo de conversação do Banco Ágil.

Executa cenários completos contra o grafo LangGraph, salva os resultados
em test_results/ e gera um relatório Markdown com avaliação detalhada.

Uso:
    python test_flows.py
"""

import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Lock

from langchain_core.messages import AIMessage, HumanMessage

from state import build_graph

# ============================================================
# Configuração
# ============================================================

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Cenários de teste
# Cada cenário representa um fluxo de atendimento distinto.
#
# Clientes disponíveis (data/clients.csv):
#   Ana Clara       12345678901  1990-05-15  score=650  limit=5000  max=8000
#   João Pedro      23456789012  1955-08-22  score=720  limit=12000 max=12000
#   Maria Fernanda  34567890123  2005-11-03  score=580  limit=2000  max=5000
#   Carlos Eduardo  45678901234  1978-02-28  score=700  limit=8000  max=12000
#   Dona Aparecida  56789012345  1945-07-10  score=600  limit=3000  max=8000
# ============================================================

SCENARIOS = [
    {
        "name": "01_auth_consulta_limite",
        "description": "Autenticação bem-sucedida + consulta de limite de crédito",
        "inputs": [
            "12345678901",                   # CPF Ana Clara
            "15/05/1990",                    # DOB correta
            "Quero consultar meu limite de crédito",
        ],
        "assertions": {
            "is_authenticated": True,
            "auth_attempts": 0,
        },
        "expected_tools": ["authenticate_user", "check_limit"],
    },
    {
        "name": "02_auth_aumento_aprovado",
        "description": "Autenticação + aumento de limite aprovado pelo score (Dona Aparecida: score=600, max=R$8000)",
        "inputs": [
            "56789012345",                   # CPF Dona Aparecida
            "10/07/1945",                    # DOB correta
            "Gostaria de aumentar meu limite de crédito",
            "Quero um limite de 6000 reais",  # Dentro do máximo permitido (R$8000)
        ],
        "assertions": {
            "is_authenticated": True,
        },
        "expected_tools": ["authenticate_user", "request_limit_increase"],
    },
    {
        "name": "03_auth_aumento_rejeitado_entrevista",
        "description": "Aumento rejeitado (score insuficiente) → entrevista → reavaliação (Maria Fernanda: score=580, max=R$5000)",
        "inputs": [
            "34567890123",                   # CPF Maria Fernanda
            "03/11/2005",                    # DOB correta
            "Preciso de um limite de 15000 reais",  # Acima do max permitido
            "Sim, quero fazer a entrevista de crédito",
            # Resposta consolidada com todos os 5 dados da entrevista
            "Trabalho com carteira assinada, ganho 4000 por mês, gasto 1200 de despesas fixas, não tenho dependentes e não tenho dívidas",
        ],
        "assertions": {
            "is_authenticated": True,
        },
        "expected_tools": ["authenticate_user", "request_limit_increase", "calculate_and_update_score"],
    },
    {
        "name": "04_auth_cambio_dolar",
        "description": "Autenticação + consulta de cotação do dólar via API externa",
        "inputs": [
            "23456789012",                   # CPF João Pedro
            "22/08/1955",                    # DOB correta
            "Quanto está o dólar hoje?",
        ],
        "assertions": {
            "is_authenticated": True,
        },
        "expected_tools": ["authenticate_user", "get_currency_rate"],
    },
    {
        "name": "05_auth_cambio_euro",
        "description": "Autenticação + consulta de euro seguida de conversão de valor",
        "inputs": [
            "45678901234",                   # CPF Carlos Eduardo
            "28/02/1978",                    # DOB correta
            "Qual a cotação do euro?",
            "Quanto seriam 500 euros em reais?",
        ],
        "assertions": {
            "is_authenticated": True,
        },
        "expected_tools": ["authenticate_user", "get_currency_rate"],
    },
    {
        "name": "06_bloqueio_3_tentativas",
        "description": "Três DOBs erradas consecutivas resultam em bloqueio da conta",
        "inputs": [
            "12345678901",                   # CPF Ana Clara (válido)
            "01/01/2000",                    # DOB errada — tentativa 1
            "01/01/2001",                    # DOB errada — tentativa 2
            "01/01/2002",                    # DOB errada — tentativa 3
        ],
        "assertions": {
            "is_authenticated": False,
            "auth_attempts": 3,
        },
        "expected_tools": ["authenticate_user"],
    },
    {
        "name": "07_cpf_inexistente",
        "description": "CPF não cadastrado no sistema — deve informar falha",
        "inputs": [
            "99999999999",                   # CPF não cadastrado
            "01/01/1990",
        ],
        "assertions": {
            "is_authenticated": False,
        },
        "expected_tools": ["authenticate_user"],
    },
    {
        "name": "08_encerramento_antecipado",
        "description": "Usuário encerra antes de se autenticar",
        "inputs": [
            "Não preciso de atendimento, pode encerrar",
        ],
        "assertions": {
            "is_authenticated": False,
            "current_agent": "ended",
        },
        "expected_tools": ["end_conversation"],
    },

    # ============================================================
    # Cenários com linguagem natural/informal (pessoa real)
    # ============================================================

    {
        "name": "09_natural_entrevista_credito_fix",
        "description": "FIX: 'entrevista de crédito' deve rotear para interview, não credit",
        "inputs": [
            "34567890123",                   # Maria Fernanda
            "03/11/2005",
            "Preciso de um limite de 15000 reais",
            "Sim, quero fazer a entrevista de crédito",
            "Trabalho com carteira assinada, ganho 4000 por mês, gasto 1200 de despesas fixas, não tenho dependentes e não tenho dívidas",
        ],
        "assertions": {
            "is_authenticated": True,
        },
        "expected_tools": ["authenticate_user", "request_limit_increase", "calculate_and_update_score"],
    },
    {
        "name": "10_natural_consulta_informal",
        "description": "Linguagem informal: 'opa', 'quanto eu tenho', gírias",
        "inputs": [
            "12345678901",
            "15/05/1990",
            "opa, quero ver quanto eu tenho de limite",
        ],
        "assertions": {
            "is_authenticated": True,
        },
        "expected_tools": ["authenticate_user", "check_limit"],
    },
    {
        "name": "11_natural_cambio_informal",
        "description": "Linguagem informal: 'e aí', 'quanto tá o dólar'",
        "inputs": [
            "23456789012",
            "22/08/1955",
            "e ai, quanto ta o dolar hoje?",
        ],
        "assertions": {
            "is_authenticated": True,
        },
        "expected_tools": ["authenticate_user", "get_currency_rate"],
    },
    {
        "name": "12_natural_entrevista_dados_separados",
        "description": "Entrevista com dados fornecidos um a um de forma conversacional",
        "inputs": [
            "45678901234",                   # Carlos Eduardo
            "28/02/1978",
            "quero melhorar meu score",
            "ganho 5000",                    # renda
            "sou CLT",                       # emprego
            "gasto uns 2000 por mês",        # despesas
            "tenho 2 dependentes",           # dependentes
            "não tenho dívidas",             # dívidas
        ],
        "assertions": {
            "is_authenticated": True,
        },
        "expected_tools": ["authenticate_user", "calculate_and_update_score"],
    },
]


# ============================================================
# Utilitários
# ============================================================

def normalize_content(content) -> str:
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


def sanitize_tags(text: str) -> str:
    """Remove tags internas de roteamento."""
    cleaned = re.sub(r"\bROTA:[A-Z_]+\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


# ============================================================
# Runner de cenário
# ============================================================

def run_scenario(scenario: dict) -> dict:
    """
    Executa um cenário turno a turno em grafo isolado (SQLite em memória)
    e retorna os resultados.

    Cada chamada constrói seu próprio grafo com banco em memória para
    permitir execução paralela sem contenção.
    """
    # Grafo isolado por cenário — sem compartilhamento de conexão SQLite
    graph, _conn = build_graph(":memory:")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    local_state = {
        "is_authenticated": False,
        "auth_attempts": 0,
        "current_user_cpf": "",
        "current_user_name": "",
        "pending_cpf": "",
        "routing_target": "",
        "interview_context": {},
        "current_agent": "triage",
    }

    turns = []
    all_tools_called = []  # acumula apenas ferramentas novas por turno
    prev_msg_count = 0     # rastreia quantas mensagens já foram processadas

    for step, user_input in enumerate(scenario["inputs"], 1):
        turn = {
            "step": step,
            "user": user_input,
            "assistant": None,
            "tools_called": [],
            "state_snapshot": None,
            "error": None,
        }

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

            # Coletar apenas ferramentas chamadas NESTE turno (mensagens novas)
            all_messages = result.get("messages", [])
            new_messages = all_messages[prev_msg_count:]
            prev_msg_count = len(all_messages)

            for msg in new_messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        turn["tools_called"].append(tc["name"])
                        all_tools_called.append(tc["name"])

            # Resposta final visível ao usuário (entre as novas mensagens)
            ai_msgs = [
                m for m in new_messages
                if isinstance(m, AIMessage) and m.content and not m.tool_calls
            ]
            if ai_msgs:
                turn["assistant"] = sanitize_tags(normalize_content(ai_msgs[-1].content))
            else:
                turn["assistant"] = "(sem resposta de texto)"

        except Exception as e:
            turn["error"] = str(e)
            turn["assistant"] = f"[ERRO: {e}]"

        turn["state_snapshot"] = {
            k: v for k, v in local_state.items() if k != "interview_context"
        }
        turns.append(turn)

    # ---- Avaliação ----
    assertion_results = {}
    for key, expected in scenario.get("assertions", {}).items():
        actual = local_state.get(key)
        assertion_results[key] = {
            "expected": expected,
            "actual": actual,
            "passed": actual == expected,
        }

    tool_checks = {
        t: {"passed": t in all_tools_called}
        for t in scenario.get("expected_tools", [])
    }

    state_passed = all(r["passed"] for r in assertion_results.values())
    tools_passed = all(r["passed"] for r in tool_checks.values())

    return {
        "scenario": scenario["name"],
        "description": scenario["description"],
        "passed": state_passed and tools_passed,
        "state_passed": state_passed,
        "tools_passed": tools_passed,
        "turns": turns,
        "all_tools_called": all_tools_called,
        "assertion_results": assertion_results,
        "tool_checks": tool_checks,
        "final_state": {k: v for k, v in local_state.items() if k != "interview_context"},
    }


# ============================================================
# Geração do relatório Markdown
# ============================================================

def generate_report(results: list, timestamp: str) -> str:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    lines = [
        "# Relatório de Testes — Banco Ágil",
        "",
        f"**Data/Hora:** {timestamp.replace('_', ' ').replace('-', ':', 2)}  ",
        f"**Total:** {total} &nbsp;|&nbsp; **Passou:** {passed} &nbsp;|&nbsp; **Falhou:** {failed}",
        "",
        "---",
        "",
        "## Resumo Geral",
        "",
        "| # | Cenário | Status | Estado | Ferramentas |",
        "|---|---------|--------|--------|-------------|",
    ]

    for i, r in enumerate(results, 1):
        status  = "✅ PASSOU" if r["passed"]       else "❌ FALHOU"
        st_icon = "✅"        if r["state_passed"] else "❌"
        tl_icon = "✅"        if r["tools_passed"] else "❌"
        lines.append(f"| {i} | `{r['scenario']}` | {status} | {st_icon} | {tl_icon} |")

    lines += ["", "---", ""]

    # Detalhamento por cenário
    for r in results:
        status = "✅ PASSOU" if r["passed"] else "❌ FALHOU"
        lines += [
            f"## {status} &mdash; `{r['scenario']}`",
            "",
            f"> {r['description']}",
            "",
        ]

        # Asserções de estado
        if r["assertion_results"]:
            lines += ["**Asserções de estado:**", ""]
            for key, val in r["assertion_results"].items():
                icon = "✅" if val["passed"] else "❌"
                lines.append(
                    f"- {icon} `{key}`: esperado **{val['expected']}**, obtido **{val['actual']}**"
                )
            lines.append("")

        # Ferramentas esperadas
        if r["tool_checks"]:
            lines += ["**Ferramentas esperadas:**", ""]
            for tool, val in r["tool_checks"].items():
                icon = "✅" if val["passed"] else "❌"
                lines.append(f"- {icon} `{tool}`")
            lines.append("")

        # Todas as ferramentas chamadas
        if r["all_tools_called"]:
            seq = " → ".join(f"`{t}`" for t in r["all_tools_called"])
            lines += [f"**Sequência de ferramentas:** {seq}", ""]

        # Transcrição
        lines += ["**Transcrição da conversa:**", ""]
        for turn in r["turns"]:
            lines.append(f"**[{turn['step']}] 👤 Usuário:** {turn['user']}")
            if turn["tools_called"]:
                tools_str = ", ".join(f"`{t}`" for t in turn["tools_called"])
                lines.append(f"*→ Ferramentas chamadas: {tools_str}*")
            resp = turn["assistant"] or ""
            # Truncar respostas longas no relatório
            preview = resp[:500] + ("..." if len(resp) > 500 else "")
            lines.append(f"**🤖 Agilito:** {preview}")
            if turn.get("error"):
                lines.append(f"⚠️ **Erro:** {turn['error']}")
            lines.append("")

        # Estado final do cenário
        fs = r["final_state"]
        lines += [
            "**Estado final:**",
            f"- `is_authenticated`: {fs.get('is_authenticated')}",
            f"- `auth_attempts`: {fs.get('auth_attempts')}",
            f"- `current_agent`: {fs.get('current_agent')}",
            f"- `current_user_name`: {fs.get('current_user_name') or '—'}",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Número de workers paralelos (ajuste conforme CPU/API disponível)
    MAX_WORKERS = int(os.getenv("TEST_WORKERS", "4"))

    print("=" * 60)
    print("  Banco Agil -- Testes de Fluxo de Conversacao")
    print(f"  Paralelo: {MAX_WORKERS} workers | SQLite em memoria")
    print("=" * 60)
    print()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # results na ordem original dos cenarios
    results = [None] * len(SCENARIOS)
    print_lock = Lock()

    def _run_and_report(idx: int, scenario: dict):
        result = run_scenario(scenario)
        status = "PASSOU" if result["passed"] else "FALHOU"
        lines = [f"[{idx+1:02d}/{len(SCENARIOS):02d}] {scenario['name']}  {status}"]
        if not result["passed"]:
            for key, val in result["assertion_results"].items():
                if not val["passed"]:
                    lines.append(f"         Estado  -- {key}: esperado={val['expected']}, obtido={val['actual']}")
            for tool, val in result["tool_checks"].items():
                if not val["passed"]:
                    lines.append(f"         Ferramenta nao chamada: {tool}")
        with print_lock:
            print("\n".join(lines), flush=True)
        return idx, result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_run_and_report, i, s): i
            for i, s in enumerate(SCENARIOS)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    # Salvar JSON bruto
    json_path = os.path.join(RESULTS_DIR, f"{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # Salvar relatório Markdown
    md_path = os.path.join(RESULTS_DIR, f"{timestamp}_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_report(results, timestamp))

    # Sumário final
    total  = len(results)
    passed = sum(1 for r in results if r["passed"])
    print()
    print("=" * 60)
    print(f"  Resultado: {passed}/{total} cenarios passaram")
    print(f"  Relatorio: {md_path}")
    print(f"  JSON:      {json_path}")
    print("=" * 60)

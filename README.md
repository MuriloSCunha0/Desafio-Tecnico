# Banco Ágil — Sistema Multi-Agente Bancário

Sistema de atendimento ao cliente para um banco digital fictício, construído com **LangGraph**, **LangChain** e **Streamlit**. Quatro agentes de IA especializados trabalham em conjunto para oferecer uma experiência de atendimento fluida e inteligente.

---

## Visão Geral

O Banco Ágil simula um assistente virtual bancário chamado **Agilito**, que opera por meio de 4 agentes de IA:

- **Triagem** — autentica o cliente (CPF + data de nascimento) e direciona para o agente correto
- **Crédito** — consulta limite, processa aumento de limite, registra solicitações
- **Entrevista** — conduz entrevista financeira e recalcula score de crédito
- **Câmbio** — consulta cotação de moedas em tempo real via API externa

O diferencial está na **transparência zero para o cliente**: as transições entre agentes são imperceptíveis — para o usuário, é uma conversa natural com um único assistente.

---

## Arquitetura do Sistema

O sistema utiliza um **StateGraph** do LangGraph, onde cada nó é um agente e as transições são controladas por uma combinação de lógica Python e análise contextual do LLM.

### Roteamento Híbrido (Python + LLM)

Uma decisão técnica importante foi implementar o roteamento em duas camadas:

1. **Python-driven routing** (camada primária): O agente de triagem usa uma máquina de estados de 4 fases para autenticação. Após autenticar, a detecção de intenção é feita por **análise de palavras-chave em Python** (`_detect_routing_target()`), que define o campo `routing_target` no estado do grafo. Isso garante roteamento determinístico e não depende do LLM para decisões de fluxo.

2. **LLM-driven routing** (fallback): Para transições entre agentes especializados (ex: entrevista → crédito), o LLM pode incluir tags internas (`ROTA:CREDITO`) que são detectadas pelo roteador e removidas antes de exibir ao cliente.

Essa abordagem surgiu de testes com modelos menores (Ollama qwen2.5:3b) que não conseguiam manter o fluxo de roteamento via LLM de forma confiável. O sistema Python garante que o fluxo funcione independentemente do modelo.

### Fases do Agente de Triagem

```
Fase A: Sem CPF       → pede CPF (end_conversation disponível)
Fase B: CPF coletado  → pede DOB (sem tools)
Fase C: CPF + DOB     → chama authenticate_user
Fase D: Autenticado   → detecta intenção via Python → routing_target
```

### Diagrama de Fluxo

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit (app.py)                  │
│              Interface de Chat do Usuário            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│            StateGraph (LangGraph — state.py)        │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌────────────┐      │
│  │ Triagem  │──→│ Crédito  │←──│ Entrevista │      │
│  │  Agent   │   │  Agent   │   │   Agent    │      │
│  └────┬─────┘   └──────────┘   └────────────┘      │
│       │                                             │
│       └────────→┌──────────┐                        │
│                 │  Câmbio  │                        │
│                 │  Agent   │                        │
│                 └──────────┘                        │
│                                                     │
│  ┌──────────────────────────────────────────┐       │
│  │     Tool Executor + SqliteSaver          │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌────────────────┐
    │clients.  │ │score_    │ │solicitacoes_   │
    │csv       │ │limite.csv│ │aumento_        │
    │          │ │          │ │limite.csv      │
    └──────────┘ └──────────┘ └────────────────┘
```

### Agentes e Ferramentas

| Agente | Ferramentas | Responsabilidade |
|--------|------------|------------------|
| **Triagem** | `authenticate_user`, `end_conversation` | Autenticação em 4 fases, roteamento por keywords |
| **Crédito** | `check_limit`, `request_limit_increase`, `end_conversation` | Consulta/aumento de limite, registro formal em CSV |
| **Entrevista** | `calculate_and_update_score`, `end_conversation` | Coleta de 5 dados financeiros, cálculo de score e atualização |
| **Câmbio** | `get_currency_rate`, `end_conversation` | Cotação de moedas via API (frankfurter.app) |

### Fluxo de Dados

1. **Autenticação** → `clients.csv` (CPF + data de nascimento)
2. **Consulta de limite** → `clients.csv` (score + current_limit)
3. **Solicitação de aumento** → `score_limite.csv` (verifica faixa) → `solicitacoes_aumento_limite.csv` (registra pedido com status)
4. **Entrevista** → fórmula ponderada → `clients.csv` (atualiza score + limite)
5. **Câmbio** → API frankfurter.app → cotação em tempo real

---

## Funcionalidades Implementadas

- ✅ **Autenticação com bloqueio progressivo** — 3 tentativas com mensagem amigável de encerramento
- ✅ **Consulta de limite e score de crédito**
- ✅ **Solicitação de aumento de limite** com registro formal em `solicitacoes_aumento_limite.csv`
- ✅ **Verificação automatizada** via tabela `score_limite.csv`
- ✅ **Entrevista de reavaliação** com coleta de 5 dados financeiros (renda, emprego, despesas, dependentes, dívidas)
- ✅ **Parser inteligente** — aceita respostas informais ("ganho dois mil" → 2000, "CLT" → formal)
- ✅ **Fórmula ponderada de score** (0-1000) conforme especificação do desafio
- ✅ **Atualização de score em `clients.csv`** com redirecionamento automático para nova análise
- ✅ **Consulta de câmbio** via API pública (frankfurter.app, sem API key)
- ✅ **Encerramento a qualquer momento** via `end_conversation`
- ✅ **Transições imperceptíveis** entre agentes
- ✅ **Tratamento de erros** (CSV, API, entrada inválida) com log em `logs/errors.log`
- ✅ **Tom acessível e didático** adaptado para todas as faixas etárias (18-80 anos)
- ✅ **Persistência de conversa** via SqliteSaver do LangGraph
- ✅ **3 providers de LLM** — Ollama (dev local), Google Gemini, Groq (free tier)
- ✅ **12 testes automatizados** cobrindo todos os fluxos, incluindo linguagem informal

---

## Fórmula de Score

```python
score = (renda_mensal / (despesas + 1)) * 30 + peso_emprego + peso_dependentes + peso_dividas
```

| Componente | Pesos |
|-----------|-------|
| `peso_renda` | 30 |
| `peso_emprego` | formal=300, autônomo=200, desempregado=0 |
| `peso_dependentes` | 0→100, 1→80, 2→60, 3+→30 |
| `peso_dividas` | não→+100, sim→-100 |

Score é clamped entre 0 e 1000.

---

## Desafios Enfrentados e Soluções

### 1. Roteamento confiável com modelos pequenos

**Problema:** Ao testar com Ollama (qwen2.5:3b) e Groq (Llama 3.3), os modelos frequentemente erravam o roteamento — chamavam ferramentas prematuramente, inventavam nomes de tools, ou pulavam etapas de autenticação.

**Solução:** Implementei um **roteamento híbrido Python + LLM**. A autenticação é controlada por uma máquina de estados de 4 fases em Python puro. O roteamento pós-autenticação usa detecção de palavras-chave (`_detect_routing_target()`) com análise contextual para respostas ambíguas ("sim", "ok"). As ferramentas são injetadas via `_make_tool_call_message()` sem depender do LLM para a decisão.

### 2. Tool injection vs tool calling

**Problema:** Modelos menores não conseguiam chamar `calculate_and_update_score` corretamente — um até inventou o nome `calculate_credit_score`.

**Solução:** Os agentes de crédito, entrevista e câmbio usam **Python-driven tool injection**: o código Python decide quando chamar cada ferramenta e constrói a `AIMessage` com `tool_calls` diretamente, sem passar pelo LLM. O LLM só é usado para gerar respostas textuais ao cliente.

### 3. Entrevista conversacional flexível

**Problema:** O cliente pode enviar todos os dados de uma vez ("ganho 4000, sou CLT, gasto 1200, sem dependentes, sem dívidas") ou responder pergunta por pergunta.

**Solução:** O `_parse_interview_fields_context()` usa duas estratégias: primeiro tenta extrair de texto combinado com regex, depois analisa pares (pergunta AI → resposta humana) em ordem cronológica, consumindo cada resposta para no máximo um campo.

### 4. Otimização de tokens para Groq free tier

**Problema:** O Groq free tier tem limites rigorosos de tokens por minuto/dia.

**Solução:** Comprimi todos os system prompts (~54% de redução de linhas) mantendo todas as regras. Para o contexto da entrevista, uso formato compacto inspirado em TOON ao invés de bullet-points verbose.

### 5. Persistência de estado

**Problema:** Manter autenticação, tentativas, contexto de entrevista e agente atual entre mensagens.

**Solução:** `SqliteSaver` do LangGraph como checkpointer + `AgentState` TypedDict com 9 campos de controle.

---

## Escolhas Técnicas e Justificativas

### Por que LangGraph?

O LangGraph foi escolhido porque oferece exatamente o que um sistema multi-agente precisa:

- **StateGraph** — permite definir cada agente como um nó com transições condicionais, criando um grafo de execução que mapeia naturalmente para o fluxo do atendimento bancário
- **Persistência nativa** — o `SqliteSaver` mantém o estado da conversa entre mensagens sem precisar de infraestrutura externa
- **Tool calling padronizado** — integração direta com `@tool` do LangChain, com controle granular sobre quais ferramentas cada agente pode acessar
- **Flexibilidade de roteamento** — arestas condicionais permitem combinar lógica Python (determinística) com decisões do LLM (adaptativas)

### Demais Escolhas

| Escolha | Por quê? |
|---------|----------|
| **LangChain `@tool`** | Decorador padronizado que funciona com qualquer LLM da família LangChain |
| **CSV como armazenamento** | Portabilidade total, sem dependências externas, ideal para o escopo do desafio |
| **Streamlit** | Interface de chat pronta com `st.chat_message`, deploy rápido |
| **frankfurter.app** | API de câmbio gratuita, sem necessidade de API key, dados confiáveis do ECB |
| **3 providers (Ollama/Google/Groq)** | Factory `get_llm()` permite testar local (Ollama) e escalar para produção (Groq/Google) |
| **Logging rotativo** | `RotatingFileHandler` (5MB, 3 backups) para diagnóstico sem overhead em disco |

---

## Tutorial de Execução

### Pré-requisitos

- **Python 3.11+**
- Uma das opções de LLM:
  - **Ollama** com modelo `llama3.1:8b` (dev local)
  - Chave de API do **Groq** (free tier funciona) — recomendado
  - Chave de API do **Google Gemini**
- **Docker** e **Docker Compose** (opcional)

### Opção 1: Execução Local

```bash
# 1. Clonar o repositório
git clone <url-do-repositório>
cd desafio-tecnico

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar ambiente
cp .env.example .env
# Edite .env com sua API key e provider

# 4. (Se usar Ollama) Baixar o modelo
ollama pull llama3.1:8b

# 5. Executar
streamlit run app.py

# 6. Acessar: http://localhost:8501
```

### Opção 2: Docker

```bash
# 1. Configurar ambiente
cp .env.example .env
# Edite .env com sua API key

# 2. Construir e subir
docker compose up --build

# 3. Acessar: http://localhost:8501
```

### Testando o Sistema

**Testes manuais no chat:**

1. **Autenticação**: CPF `12345678901`, data `15/05/1990` (Ana Clara Silva)
2. **Consulta de crédito**: "Quero ver meu limite"
3. **Aumento de limite**: Peça R$ 15.000 para ver rejeição (score baixo)
4. **Entrevista**: Aceite a entrevista, responda as 5 perguntas
5. **Câmbio**: "Quanto está o dólar?"
6. **Bloqueio**: Erre 3 vezes os dados para ver o bloqueio

**Testes automatizados (12 cenários):**

```bash
python test_flows.py
# Relatório gerado em test_results/
```

---

## Estrutura do Projeto

```
├── app.py                    # Frontend Streamlit (UI de chat)
├── agents/                   # Pacote de Agentes de IA
│   ├── __init__.py           # Exportação do pacote
│   ├── core.py               # Utilitários, LLM factory, parsing compartilhado
│   ├── triage.py             # Agente de Triagem (4 fases de autenticação)
│   ├── credit.py             # Agente de Crédito (consulta/aumento de limite)
│   ├── interview.py          # Agente de Entrevista (reavaliação de score)
│   └── forex.py              # Agente de Câmbio (cotação de moedas)
├── state.py                  # StateGraph, AgentState, roteamento, tool executor
├── tools.py                  # 6 tools (@tool LangChain)
├── db_utils.py               # CRUD para arquivos CSV
├── logger.py                 # Logging rotativo (logs/errors.log)
├── test_flows.py             # 12 cenários de teste automatizados
├── requirements.txt          # Dependências Python
├── .env.example              # Template de configuração
├── Dockerfile                # Imagem Docker
├── docker-compose.yml        # Compose com volumes
├── README.md                 # Este arquivo
└── data/
    ├── clients.csv           # Clientes (CPF, nome, nascimento, score, limite)
    ├── score_limite.csv      # Tabela score → limite máximo
    └── solicitacoes_aumento_limite.csv  # Log de solicitações de aumento
```

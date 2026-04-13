---
title: ESG Auditor
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
tags: [streamlit, esg, langgraph, claude, audit, sr11-7]
license: mit
pinned: false
short_description: SR 11-7 compliant multi-agent ESG portfolio auditor
---

# ESG Risk Compliance Auditor
### Multi-Agent AI System — SR 11-7 Compliant

A production-grade multi-agent ESG portfolio auditor that automates ESG compliance analysis for financial services. The system coordinates three LLM-backed agents — Advisor, Analyst, and Client — through a LangGraph orchestration layer, pulling real-time data from financial APIs, SEC filings, news sentiment analysis, and a regulatory knowledge base to produce structured ESG audit reports.

---

## Author

**Nathon Chavez**

---

## Repository Structure

```
ESGRiskComplianceAuditor/
├── src/
│   └── esg_auditor/
│       ├── app.py                     # Streamlit entry point
│       ├── config.py                  # pydantic-settings configuration
│       ├── agents/
│       │   ├── graph.py               # LangGraph StateGraph + build_graph()
│       │   ├── advisor_agent.py       # Supervisor orchestrator
│       │   ├── analyst_agent.py       # ReAct sub-graph with 6 tools
│       │   ├── client_agent.py        # Structured output ClientProfile
│       │   └── prompts.py             # System prompts for all three agents
│       ├── core/
│       │   ├── schemas.py             # Pydantic v2: ESGReport, AuditRequest, ClientProfile
│       │   ├── state.py               # LangGraph AgentState TypedDict
│       │   ├── exceptions.py          # Typed exception hierarchy
│       │   ├── circuit_breaker.py     # CircuitBreaker state machine
│       │   └── contracts.py           # Inter-agent handoff validator
│       ├── tools/
│       │   ├── marketaux.py           # News sentiment API
│       │   ├── finnhub_tools.py       # ESG scores (Primary)
│       │   ├── yfinance_tools.py      # ESG scores (Fallback)
│       │   ├── sec_edgar.py           # SEC full-text search
│       │   ├── qdrant_search.py       # Vector search for regulatory docs
│       │   └── finbert.py             # FinBERT dual-model NLP inference
│       ├── rag/
│       │   ├── chunker.py             # Section-aware regulatory text chunking
│       │   ├── embedder.py            # bge-small-en-v1.5 embedding pipeline
│       │   └── ingest.py              # Document ingestion to Qdrant
│       ├── ui/
│       │   ├── chat.py                # Streamlit chat interface
│       │   ├── dashboard.py           # ESG gauges and metric cards
│       │   └── components.py          # Audit trail renderer
│       └── eval/
│           ├── ragas_eval.py          # RAGAS evaluation pipeline
│           ├── finbert_eval.py        # FinBERT accuracy on financial phrasebank
│           ├── latency_benchmark.py   # End-to-end latency p50/p95/p99
│           └── datasets/
│               ├── esg_qa.json        # Ground truth evaluation dataset
│               └── metrics_baseline.json  # Populated by evaluation notebook
├── scripts/
│   └── ingest_regulatory_docs.py     # One-time Qdrant seed ingest CLI
├── notebooks/
│   └── esg_auditor_evaluation.ipynb  # Full evaluation notebook (8 figures)
├── tests/
│   ├── conftest.py
│   ├── unit/                          # 59 unit tests — no real API calls
│   └── integration/                   # RAG pipeline + E2E graph tests
├── docs/
│   └── MODEL_CARD.md                  # SR 11-7 compliance document
├── Dockerfile                         # HF Spaces Docker SDK container
├── pyproject.toml                     # PEP 621 dependency spec
├── requirements.txt                   # Docker-compatible pinned deps
├── Makefile                           # install / lint / test / run
├── .env.example                       # Credential template (no real values)
├── .github/
│   └── workflows/
│       ├── ci.yml                     # Lint + unit tests on PR/push
│       ├── deploy.yml                 # Sync to HF Spaces on main push
│       └── keepalive.yml              # 12h ping to prevent HF sleep
└── README.md
```

---

## Architecture

The system uses an **orchestration pattern** (not choreography): a central LangGraph supervisor manages all agent coordination, state versions are immutable snapshots at every handoff, and every agent call is guarded by a circuit breaker.

```
User → Streamlit UI → LangGraph StateGraph
                              │
                        Advisor Agent (Claude Sonnet)
                         ↙              ↘
              Client Agent          Analyst Agent (ReAct)
            (ClientProfile)              │
                                ┌────────┼────────┐
                           Marketaux   SEC    Finnhub
                           FinBERT   Qdrant  yfinance
                                         │
                                   Qdrant Cloud
                                  (Regulatory RAG)
```

**Four distributed systems patterns** baked in from day one:
1. **Immutable state versioning** — `state_version` increments at every agent handoff
2. **Circuit breaker** — `CircuitBreaker` CLOSED/OPEN/HALF_OPEN state machine on every external call
3. **Data contracts** — Pydantic schema validation at every inter-agent boundary
4. **Saga compensation** — `executed_agents` list enables rollback on failure

---

## Credentials Required

> **⚠️ The following API keys are required to run the system. None are included in this repository.**

| Service | Purpose | Free Tier |
|---|---|---|
| [Anthropic](https://console.anthropic.com) | Claude Sonnet LLM backbone | $5 credit |
| [Qdrant Cloud](https://cloud.qdrant.io) | Vector store for regulatory docs | 1GB free |
| [Marketaux](https://www.marketaux.com) | ESG news sentiment | 100 req/day |
| [Finnhub](https://finnhub.io) | ESG scores (primary) | Free tier |
| [LangSmith](https://smith.langchain.com) | SR 11-7 audit traces | 5,000 traces/month |

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
# Edit .env with your real credentials
```

---

## Installation

### Option A — Local with pip

```bash
git clone https://github.com/CrimsonCovenant/ESGRiskComplianceAuditor.git
cd ESGRiskComplianceAuditor
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env               # Fill in your credentials
pre-commit install
```

### Option B — Docker (matches HF Spaces production)

```bash
git clone https://github.com/CrimsonCovenant/ESGRiskComplianceAuditor.git
cd ESGRiskComplianceAuditor
cp .env.example .env               # Fill in your credentials
docker build -t esg-auditor .
docker run -p 7860:7860 --env-file .env esg-auditor
# Open http://localhost:7860
```

### Key dependencies

| Package | Version | Purpose |
|---|---|---|
| `langgraph` | ≥0.3.24 | Multi-agent orchestration |
| `langchain-anthropic` | ≥0.3.12 | Claude API integration |
| `qdrant-client[fastembed]` | ≥1.13.3 | Vector store + embeddings |
| `transformers` | ≥4.48.0 | FinBERT NLP models |
| `torch` | ≥2.5.1 | CPU inference for FinBERT |
| `streamlit` | ≥1.42.0 | Web UI |
| `pydantic` | ≥2.10.6 | Data validation + schemas |
| `langsmith` | ≥0.3.12 | SR 11-7 audit tracing |

---

## Running the Application

```bash
make run
# Opens Streamlit on http://localhost:8501
```

The app has three pages:
- **Chat** — submit ESG audit queries, view agent trace per response
- **Dashboard** — ESG gauge charts and metric cards (populated after an audit)
- **Audit Trail** — SR 11-7 compliant 12-field log with JSON export

### Example queries

```
Audit Microsoft (MSFT) for ESG compliance
What are the ESG risks for ExxonMobil (XOM)?
Compare ESG profiles for defense contractors LMT and RTX
```

---

## Seeding the Regulatory Knowledge Base

The Qdrant vector store must be seeded before the first audit. Run once after setup:

```bash
python scripts/ingest_regulatory_docs.py
# Expected output:
# ✓ Ingested: SR-11-7-FRB-2011 (2 chunks)
# ✓ Ingested: TCFD-Recommendations-2017 (2 chunks)
# ✓ Ingested: SEC-ClimateRule-2024 (2 chunks)
# ✓ Ingested: FINRA-2026-RAOR-GenAI (2 chunks)
# ✓ Ingested: EU-CSRD-2022 (2 chunks)
# Total: 10 chunks ingested into esg_regulatory_docs
```

Verify with `--dry-run` first:

```bash
python scripts/ingest_regulatory_docs.py --dry-run
```

---

## Pipeline Stages

| # | Stage | Description |
|---|---|---|
| 1 | User Input | Query submitted via Streamlit chat interface |
| 2 | State Init | `make_initial_state()` creates zeroed `AgentState` with `state_version=0` |
| 3 | Advisor Pass 1 | Advisor calls `get_client_profile` AND `consult_analyst` in parallel |
| 4 | Client Agent | Generates structured `ClientProfile` via `with_structured_output` |
| 5 | Analyst — ESG Scores | Calls Finnhub → falls back to yfinance on 403 |
| 6 | Analyst — News | Fetches ESG headlines via Marketaux |
| 7 | Analyst — NLP | Runs FinBERT sentiment + ESG classification on headlines |
| 8 | Analyst — Filings | Searches SEC EDGAR for 10-K/10-Q ESG disclosures |
| 9 | Analyst — Regulatory | Queries Qdrant for top-5 regulatory document chunks |
| 10 | Advisor Pass 2 | Synthesises all agent outputs into structured 6-section report |
| 11 | Output | ESG report rendered in UI; audit entry appended to session log |
| 12 | Tracing | LangSmith captures all 12 SR 11-7 required data elements |

---

## Development

```bash
make lint      # ruff check src/ tests/
make test      # pytest tests/unit/ -v
make run       # streamlit run src/esg_auditor/app.py
```

### Running the evaluation suite

```bash
# Seed Qdrant first (see above), then:

# RAGAS — RAG quality (Faithfulness, Context Precision, Context Recall)
python -m esg_auditor.eval.ragas_eval

# FinBERT — accuracy on financial phrasebank benchmark
python -m esg_auditor.eval.finbert_eval

# Latency — p50/p95/p99 over 10 end-to-end graph invocations
# Requires a real ANTHROPIC_API_KEY
python -m esg_auditor.eval.latency_benchmark

# Full evaluation notebook with 8 scientific figures
jupyter notebook notebooks/esg_auditor_evaluation.ipynb
```

---

## Evaluation Results

### RAGAS (RAG Quality — target ≥ 0.8)

| Metric | Score | Target |
|---|---|---|
| Faithfulness | [run eval to populate] | ≥ 0.8 |
| Context Precision | [run eval to populate] | ≥ 0.8 |
| Context Recall | [run eval to populate] | ≥ 0.8 |

### FinBERT (financial phrasebank benchmark)

| Metric | Score | Benchmark |
|---|---|---|
| Accuracy | [run eval to populate] | ~0.86 published |
| Macro F1 | [run eval to populate] | — |
| Macro PR-AUC | [run eval to populate] | — |

### Latency (end-to-end graph invocation)

| Percentile | Time |
|---|---|
| p50 | [run benchmark to populate] |
| p95 | [run benchmark to populate] |
| p99 | [run benchmark to populate] |

> Run `notebooks/esg_auditor_evaluation.ipynb` to populate all metrics and export `src/esg_auditor/eval/datasets/metrics_baseline.json`.

---

## Module Reference

### `agents/graph.py`

| Function | Inputs | Outputs | Description |
|---|---|---|---|
| `build_graph(settings)` | Optional Settings | CompiledGraph | Assembles and compiles the full LangGraph StateGraph |
| `make_initial_state(user_message)` | str | AgentState | Creates zeroed AgentState for a new conversation thread |
| `configure_tracing(settings)` | Settings | — | Sets LangSmith env vars — call once at app startup |
| `_get_advisor_tools(settings)` | Settings | list[tool] | Builds @tool wrappers for sub-agents with settings injected |

### `agents/analyst_agent.py`

| Function | Inputs | Outputs | Description |
|---|---|---|---|
| `run_analyst(research_query, settings)` | str, Settings | str | Runs ReAct sub-graph with all 6 ESG research tools |

### `agents/client_agent.py`

| Function | Inputs | Outputs | Description |
|---|---|---|---|
| `generate_client_profile(description, settings)` | str, Settings | str | Generates validated ClientProfile JSON via structured output |

### `rag/chunker.py`

| Function | Inputs | Outputs | Description |
|---|---|---|---|
| `chunk_regulatory_document(text, metadata)` | str, dict | list[Document] | Section-aware chunking with metadata validation |

### `eval/ragas_eval.py`

| Function | Inputs | Outputs | Description |
|---|---|---|---|
| `run_evaluation(eval_data, settings)` | dict, Settings | dict | Faithfulness, Context Precision, Context Recall scores |

### `eval/finbert_eval.py`

| Function | Inputs | Outputs | Description |
|---|---|---|---|
| `evaluate_finbert_accuracy(settings, full)` | Settings, bool | dict | Accuracy + F1 + PR-AUC on financial phrasebank |

### `eval/latency_benchmark.py`

| Function | Inputs | Outputs | Description |
|---|---|---|---|
| `run_latency_benchmark(n_runs, settings)` | int, Settings | dict | p50/p95/p99 latency over N full graph invocations |

---

## Regulatory Compliance

This system is designed to satisfy SR 11-7 model risk management requirements. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for the full compliance document.

| Requirement | Framework | Status |
|---|---|---|
| Model documentation | SR 11-7 / OCC 2011-12 | ✅ MODEL_CARD.md |
| Fiduciary duty | SEC Reg BI | ✅ Advisory-only mode |
| Suitability | FINRA Rule 2111 | ✅ Client agent profiling |
| Know your customer | FINRA Rule 2090 | ✅ Client agent intake |
| Communication logging | FINRA / SEC 17a-4 | ✅ LangSmith traces |
| Independent validation | SR 11-7 Pillar 2 | ⏳ Scheduled |
| High-risk conformity | EU AI Act | ⏳ Assessment pending |

Every agent interaction logs 12 SR 11-7 required data elements: timestamp, session ID, user input, agent invoked, tool selected, retrieval sources, model version, reasoning chain, final output, confidence indicators, policy checks, and approval status.

---

## Reproducibility Notes

- All Pydantic schemas are versioned — `ESGReport`, `AuditRequest`, `ClientProfile` define the exact input/output contract at every agent boundary.
- LangGraph `InMemorySaver` checkpointing preserves full conversation state per `thread_id`. Set `thread_id` explicitly to replay a conversation.
- FinBERT models load once via `@lru_cache` and persist for the container lifetime. First inference on a cold container takes 15-30 seconds.
- Qdrant collection name is configured via `COLLECTION_NAME` in `.env`. The seed corpus produces 10 chunks across 5 regulatory documents.
- All evaluation metrics are deterministic given the same ground truth dataset (`esg_qa.json`) and the same Claude model version. Model version is logged in every LangSmith trace.
- The `.github/workflows/keepalive.yml` pings both the HF Space and Qdrant cluster every 12 hours to prevent the HF 48-hour sleep and Qdrant 4-week cluster deletion.

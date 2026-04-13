# MODEL CARD — ESG Portfolio Auditor

> **SR 11-7 Pillar 1 (Development)** — This document satisfies the model documentation
> requirement of SR 11-7 § III.A for quantitative models and AI/ML systems used in
> financial institutions.

---

## 1. Model Details

| Field | Value |
|---|---|
| **System name** | Multi-Agent ESG Portfolio Auditor |
| **System type** | Multi-agent LLM system with RAG pipeline |
| **Architecture** | LangGraph 1.1.6 StateGraph with 3 agents (Advisor, Analyst, Client) |
| **LLM backbone** | Claude Sonnet (langchain-anthropic 1.4.0) |
| **Embedding model** | BAAI/bge-small-en-v1.5 (384-dim, via fastembed) |
| **Sentiment models** | ProsusAI/finbert (sentiment), yiyanghkust/finbert-esg (ESG classification) |
| **Vector store** | Qdrant Cloud 1.17.1 (qdrant-client) |
| **Model risk tier** | Tier 2 (Medium) — advisory output, human-in-the-loop required |
| **Version** | 0.1.0 |
| **Last validated** | 2026-04-12 |
| **Owner** | ESG Auditor Dev Team |

---

## 2. Intended Use

### Primary use case
Generate ESG compliance audit reports for publicly traded companies by
aggregating real-time financial data, regulatory document retrieval, and
sentiment analysis into a structured advisory output.

### Intended users
- Model risk management teams performing SR 11-7 compliance assessments
- ESG analysts conducting preliminary portfolio screening
- Compliance officers reviewing regulatory alignment

### Out of scope
- Automated trade execution or portfolio rebalancing
- Definitive regulatory compliance certification
- Use as sole basis for investment decisions without human review

---

## 3. Data Sources

### RAG Corpus (Qdrant)
| Source ID | Document Type | Jurisdiction | Date |
|---|---|---|---|
| SR-11-7-FRB-2011 | SR-Letter | US-Federal | 2011-04-04 |
| TCFD-Recommendations-2017 | TCFD | Global | 2017-06-29 |
| SEC-ClimateRule-2024 | SEC-Rule | US-Federal | 2024-03-06 |
| FINRA-2026-RAOR-GenAI | FINRA-Guidance | US-Federal | 2026-01-15 |
| EU-CSRD-2022 | CSRD | EU | 2022-12-14 |

### Real-Time Data APIs
| Source | Purpose | Status |
|---|---|---|
| Marketaux | ESG news headlines and sentiment | Active |
| Finnhub | ESG scores, company profile | Limited — Free tier returns 401 on ESG endpoint; premium key required |
| yfinance | Sustainability scores, financial data | Active (staleness caveat — see §6) |
| SEC EDGAR | 10-K/10-Q filing retrieval | Active |

---

## 4. Architecture

```
User Query
    ↓
┌─────────────────┐
│   Advisor        │  ← Supervisor agent (orchestrates routing)
│   (LangGraph)    │
└────┬───────┬─────┘
     │       │
     ↓       ↓
┌────────┐ ┌────────┐
│Analyst │ │Client  │
│(ReAct) │ │(Struct)│
└────────┘ └────────┘
     │
     ↓
┌──────────────────────────┐
│ Tools: Marketaux, EDGAR, │
│ yfinance, Finnhub,       │
│ FinBERT, Qdrant Search   │
└──────────────────────────┘
```

- **Advisor**: Supervisor agent that routes between Analyst and Client agents
  based on query type and conversation state
- **Analyst**: Nested ReAct sub-graph with access to 6 data tools, iteration
  guards (max 10), and tool-calling loop
- **Client**: Structured output agent producing Pydantic-validated ClientProfile
- **Circuit Breaker**: CLOSED/OPEN/HALF_OPEN state machine with saga compensation
  for graceful degradation on tool failures

---

## 5. Performance Metrics (SR 11-7 Pillar 2)

| Metric | Value | Target | How to Run |
|---|---|---|---|
| RAGAS Faithfulness | [pending] | ≥ 0.8 | `python -m esg_auditor.eval.ragas_eval` |
| RAGAS Context Precision | [pending] | ≥ 0.8 | `python -m esg_auditor.eval.ragas_eval` |
| RAGAS Context Recall | [pending] | ≥ 0.8 | `python -m esg_auditor.eval.ragas_eval` |
| FinBERT Sentiment Accuracy | [pending] | ≥ 0.85 | `python -m esg_auditor.eval.finbert_eval` |
| Latency p50 | [pending] | < 30s | `python -m esg_auditor.eval.latency_benchmark` |
| Latency p95 | [pending] | < 90s | `python -m esg_auditor.eval.latency_benchmark` |
| Latency p99 | [pending] | < 120s | `python -m esg_auditor.eval.latency_benchmark` |

### Unit Test Coverage
- 65 unit tests passing (Phases 1-5)
- 1 end-to-end integration test against real Claude API
- All evaluation pipelines tested with mocked LLM/model calls

---

## 6. Limitations and Known Risks

### Confirmed Production Limitations

1. **Finnhub ESG endpoint returns 401 on free tier** — The Finnhub ESG scores
   endpoint requires a premium API key. On free tier, `get_finnhub_esg` returns
   an error string. Mitigation: yfinance sustainability scores serve as fallback.

2. **Qdrant free tier auto-deletes after 4 weeks of inactivity** — Qdrant Cloud
   free tier clusters are deleted after prolonged inactivity. Mitigation:
   `keepalive.yml` GitHub Actions workflow pings the cluster every 12 hours.

3. **yfinance sustainability data staleness** — Sustainalytics data accessed via
   yfinance may be stale; the last update timestamp is unclear. All yfinance ESG
   outputs include a freshness warning in the tool return string.

4. **FinBERT cold start latency** — First inference takes 15-30 seconds for model
   download and initialization. Mitigated by `@lru_cache` on the model factory.

5. **transformers torchvision warnings** — Streamlit's file watcher triggers
   ModuleNotFoundError for torchvision in transformers 5.x image processing modules.
   These warnings are harmless and do not affect FinBERT functionality.

6. **LangSmith free tier trace retention** — Free tier retains 5,000 traces/month
   with 14-day retention. For SR 11-7 compliance requiring longer retention,
   supplement with JSON audit log export from the Audit Trail page.

7. **512-token context limit on FinBERT** — Texts exceeding 512 tokens are
   truncated by the tokenizer, potentially losing relevant ESG signals in
   longer documents.

---

## 7. Ethical Considerations

- **Advisory-only output**: All system outputs include the disclaimer
  "This output is advisory only and does not constitute investment advice.
  Human review is required before any action is taken."
- **No personal data processing**: The system does not process PII. Client
  profiles use synthetic or aggregated data only.
- **Bias in ESG ratings**: ESG scores from third-party providers reflect
  the methodological biases of those providers. Multiple data sources are
  cross-referenced to mitigate single-provider bias.
- **Temporal bias in FinBERT**: Model training data has a temporal cutoff;
  recent regulatory terminology may not be well-represented.

---

## 8. Maintenance Schedule

| Activity | Frequency | Owner |
|---|---|---|
| RAGAS evaluation run | Monthly | Model Risk Team |
| Qdrant corpus refresh | Quarterly | ESG Auditor Dev Team |
| LangSmith trace review | Weekly | Compliance Officer |
| Model Card update | On material change | ESG Auditor Dev Team |
| Dependency security scan | On PR (CI) | Automated via GitHub Actions |
| Circuit breaker threshold review | Quarterly | Model Risk Team |

---

## 9. Traceability — SR 11-7 Data Elements

Twelve data elements are captured per agent interaction to satisfy SR 11-7
Pillar 3 (Governance) traceability requirements:

| # | Data Element | Source | Storage |
|---|---|---|---|
| 1 | Timestamp (ISO 8601 UTC) | `build_audit_entry()` in `ui/components.py` | Session audit_log |
| 2 | Session/thread ID | uuid4 per Streamlit session | Session audit_log |
| 3 | User input | Full query text in messages | Session state + LangSmith |
| 4 | Agent invoked | `current_agent` field in AgentState | Session audit_log |
| 5 | Tool selected + parameters | LangSmith ToolNode traces | LangSmith project |
| 6 | Retrieval sources | `source_id` in `search_regulatory_docs` output | LangSmith + response |
| 7 | LLM model version | Logged in LangSmith trace metadata | LangSmith project |
| 8 | Reasoning chain | Intermediate AIMessages in state | LangSmith project |
| 9 | Final output | Last AIMessage content | Session state + LangSmith |
| 10 | Confidence indicators | FinBERT scores, explicit confidence in prompts | Response text |
| 11 | Policy checks | ADVISOR_SYSTEM_PROMPT guardrails | Prompt templates |
| 12 | Approval status | "Advisory only — human review required" | All outputs |

---

## 10. Regulatory Compliance Mapping

| Requirement | Framework | Status | Evidence |
|---|---|---|---|
| Model documentation | SR 11-7 § III.A | Complete | This MODEL_CARD.md |
| Independent validation | SR 11-7 § III.B | Scheduled | RAGAS pipeline built, baseline pending |
| Governance and controls | SR 11-7 § III.C | Complete | Audit Trail page, LangSmith traces |
| Fiduciary duty | SEC Reg BI | Enforced | Advisory-only mode in all prompts |
| Suitability | FINRA 2111 | Active | Client agent generates risk profile |
| KYC | FINRA 2090 | Active | Client agent intake |
| Communication logging | FINRA/SEC 17a-4 | Active | LangSmith traces + audit_log JSON export |
| High-risk AI conformity | EU AI Act | Pending | Conformity assessment not yet performed |
| Sustainability reporting | EU CSRD | Reference | CSRD summary in RAG corpus |
| Climate disclosure | SEC Climate Rule | Reference | SEC Climate Rule summary in RAG corpus |

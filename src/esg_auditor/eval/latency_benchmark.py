"""
Module: eval.latency_benchmark
Purpose: End-to-end latency benchmark for the LangGraph agent pipeline.
SR 11-7 Relevance: Pillar 2 (Validation) — p50/p95/p99 latency metrics
    are required in MODEL_CARD Section 5 and establish SLA baselines for
    production monitoring and incident response.
Owner: ESG Auditor Dev Team
Last Modified: 2026-04-12
"""

from __future__ import annotations

import statistics
import time

from esg_auditor.agents.graph import build_graph, make_initial_state
from esg_auditor.config import Settings, get_settings
from esg_auditor.core.exceptions import (
    AgentRoutingError,
    ConfigurationError,
)

BENCHMARK_QUERY: str = (
    "What is the ESG risk level for Apple (AAPL)? "
    "Answer in one sentence only."
)


def run_latency_benchmark(
    n_runs: int = 10,
    settings: Settings | None = None,
) -> dict[str, float | int | str]:
    """Benchmark end-to-end graph invocation latency.

    SR 11-7 Pillar 2: p50/p95/p99 latency metrics are required
    in MODEL_CARD Section 5 and establish SLA baselines for
    production monitoring and incident response.

    Args:
        n_runs: Number of invocations to time. Default 10.
        settings: Application settings. Uses get_settings()
            if None.

    Returns:
        Dict with p50_ms, p95_ms, p99_ms, mean_ms, min_ms,
        max_ms, n_runs, query.

    Raises:
        ConfigurationError: If API key is a test placeholder.
        AgentRoutingError: If any benchmark run fails.
    """
    if settings is None:
        settings = get_settings()

    if settings.anthropic_api_key.startswith("test-"):
        raise ConfigurationError(
            "Latency benchmark requires a real "
            "ANTHROPIC_API_KEY. Set it in .env "
            "before running."
        )

    graph = build_graph(settings)
    times_ms: list[float] = []

    for i in range(n_runs):
        thread_id = f"benchmark-{i:04d}"
        start = time.perf_counter()
        try:
            graph.invoke(
                make_initial_state(BENCHMARK_QUERY),
                config={
                    "configurable": {
                        "thread_id": thread_id,
                    }
                },
            )
        except Exception as exc:
            raise AgentRoutingError(
                f"Benchmark run {i} failed: {exc}"
            ) from exc
        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000
        times_ms.append(elapsed_ms)

    quantiles = statistics.quantiles(
        times_ms, n=100
    )
    return {
        "p50_ms": round(quantiles[49], 1),
        "p95_ms": round(quantiles[94], 1),
        "p99_ms": round(quantiles[98], 1),
        "mean_ms": round(
            statistics.mean(times_ms), 1
        ),
        "min_ms": round(min(times_ms), 1),
        "max_ms": round(max(times_ms), 1),
        "n_runs": n_runs,
        "query": BENCHMARK_QUERY,
    }

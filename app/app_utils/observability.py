# MediGuide — Custom Observability
# Sits on top of the scaffold's telemetry.py foundation.
# Adds clinical-specific metrics: confidence scores, red flag frequency,
# retrieval latency per agent, and token cost tracking.

import logging
import time
from contextlib import contextmanager
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

# ── Tracer and Meter ──────────────────────────────────────────────────────────
# These use the global OpenTelemetry providers set up by the ADK + Agent Runtime.
# In local dev they write to stdout. In Agent Runtime they export to Cloud Trace
# and Cloud Monitoring automatically via the OTEL exporters Agent Runtime injects.

tracer = trace.get_tracer("mediguide")
meter = metrics.get_meter("mediguide")

# ── Custom Metrics ────────────────────────────────────────────────────────────

# Histogram: how long each specialist agent's retrieval takes (ms)
retrieval_latency = meter.create_histogram(
    name="mediguide.retrieval_latency_ms",
    description="Latency of Vertex AI Search retrieval per specialist agent",
    unit="ms",
)

# Counter: how many queries triggered a red flag warning
red_flag_counter = meter.create_counter(
    name="mediguide.red_flags_detected",
    description="Number of queries where a clinical red flag was surfaced",
)

# Histogram: synthesizer confidence scores (0.0–1.0)
confidence_score = meter.create_histogram(
    name="mediguide.confidence_score",
    description="Synthesizer confidence scores for responses",
    unit="1",
)

# Counter: total Gemini API calls across all agents
llm_call_counter = meter.create_counter(
    name="mediguide.llm_calls_total",
    description="Total Gemini API calls made by all agents",
)

# Histogram: end-to-end query latency (ms)
query_latency = meter.create_histogram(
    name="mediguide.query_latency_ms",
    description="End-to-end latency from query receipt to final response",
    unit="ms",
)

# ── Context Managers ──────────────────────────────────────────────────────────

@contextmanager
def trace_agent_call(agent_name: str, query_preview: str = ""):
    """Context manager that creates a Cloud Trace span for an agent call.

    Usage:
        with trace_agent_call("diagnosis_agent", query[:50]):
            result = diagnosis_agent.run(query)
    """
    with tracer.start_as_current_span(f"mediguide.{agent_name}") as span:
        span.set_attribute("agent.name", agent_name)
        span.set_attribute("query.preview", query_preview[:100])
        start = time.time()
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            elapsed_ms = (time.time() - start) * 1000
            span.set_attribute("duration_ms", round(elapsed_ms, 2))
            retrieval_latency.record(
                elapsed_ms,
                attributes={"agent": agent_name},
            )


@contextmanager
def trace_query(session_id: str = ""):
    """Context manager for end-to-end query tracing.

    Usage:
        with trace_query(session_id) as span:
            response = orchestrator.run(query)
    """
    with tracer.start_as_current_span("mediguide.query") as span:
        span.set_attribute("session.id", session_id)
        start = time.time()
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            elapsed_ms = (time.time() - start) * 1000
            query_latency.record(elapsed_ms)
            span.set_attribute("query.duration_ms", round(elapsed_ms, 2))


# ── Event Recording ───────────────────────────────────────────────────────────

def record_confidence(score: float, query_type: str = "general") -> None:
    """Record synthesizer confidence score as a metric and span event.

    Call this from the synthesizer agent after parsing the confidence
    from its response.

    Args:
        score: Float 0.0–1.0 from synthesizer output
        query_type: One of 'drug_interaction', 'treatment', 'diagnosis', 'patient_ed'
    """
    confidence_score.record(
        score,
        attributes={"query_type": query_type},
    )
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("response.confidence", score)
        if score < 0.4:
            span.add_event(
                "low_confidence_response",
                attributes={"score": score, "query_type": query_type},
            )


def record_red_flag(flag_type: str, agent: str = "diagnosis_agent") -> None:
    """Record that a clinical red flag was detected.

    Call this when the diagnosis agent or synthesizer surfaces
    an emergency indicator.

    Args:
        flag_type: e.g. 'stemi', 'septic_shock', 'angioedema', 'suicidal_ideation'
        agent: which agent detected it
    """
    red_flag_counter.add(
        1,
        attributes={"flag_type": flag_type, "agent": agent},
    )
    span = trace.get_current_span()
    if span.is_recording():
        span.add_event(
            "red_flag_detected",
            attributes={"flag_type": flag_type, "agent": agent},
        )
    logger.warning(
        "RED FLAG detected",
        extra={"flag_type": flag_type, "agent": agent},
    )


def record_llm_call(
    agent_name: str,
    model: str = "gemini-2.5-flash",
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> None:
    """Record a Gemini API call with optional token counts.

    Args:
        agent_name: Name of the agent making the call
        model: Model identifier
        input_tokens: Prompt token count (if available)
        output_tokens: Completion token count (if available)
    """
    attrs = {"agent": agent_name, "model": model}
    llm_call_counter.add(1, attributes=attrs)

    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.agent", agent_name)
        if input_tokens:
            span.set_attribute("llm.input_tokens", input_tokens)
        if output_tokens:
            span.set_attribute("llm.output_tokens", output_tokens)
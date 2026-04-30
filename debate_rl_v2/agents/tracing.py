"""Observability & Tracing — inspired by LangSmith & OpenTelemetry.

Provides structured event logging and tracing for debugging,
monitoring, and post-hoc analysis of multi-agent debates.

Key ideas borrowed:
  - **LangSmith/LangFuse**: Trace trees with parent/child spans
  - **OpenTelemetry**: Structured attributes on spans
  - **AutoGen**: Runtime logging with structured events
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from debate_rl_v2.logging_config import get_logger

_trace_logger = get_logger("agents.tracing")


class EventLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class TraceEvent:
    """A single trace event in the debate execution."""

    name: str
    level: EventLevel = EventLevel.INFO
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_span_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level.value,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }


class Span:
    """A timed execution span (context manager).

    Usage::

        with tracer.span("llm_call", agent="proposer") as s:
            response = client.chat(...)
            s.set("tokens", response.usage["total_tokens"])
    """

    def __init__(self, tracer: "DebateTracer", name: str,
                 parent_span_id: str = "", **attributes: Any) -> None:
        self._tracer = tracer
        self._name = name
        self._parent = parent_span_id
        self._attrs = attributes
        self._span_id = uuid.uuid4().hex[:8]
        self._start: float = 0

    def set(self, key: str, value: Any) -> None:
        """Set an attribute on this span."""
        self._attrs[key] = value

    def __enter__(self) -> "Span":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration = (time.time() - self._start) * 1000
        if exc_type:
            self._attrs["error"] = str(exc_val)
            self._attrs["error_type"] = exc_type.__name__
        self._tracer._record(TraceEvent(
            name=self._name,
            level=EventLevel.ERROR if exc_type else EventLevel.INFO,
            timestamp=self._start,
            duration_ms=duration,
            attributes=self._attrs,
            trace_id=self._tracer._trace_id,
            span_id=self._span_id,
            parent_span_id=self._parent,
        ))


class DebateTracer:
    """Structured tracing for debate execution.

    Records all events (LLM calls, tool executions, state changes)
    as a trace tree that can be exported for analysis.

    Parameters
    ----------
    enabled : bool
        Whether tracing is active.
    console_output : bool
        Print events to console in real-time.
    export_path : str | None
        Auto-export trace to JSON on close.
    """

    def __init__(
        self,
        enabled: bool = True,
        console_output: bool = False,
        export_path: Optional[str] = None,
    ) -> None:
        self.enabled = enabled
        self.console_output = console_output
        self.export_path = export_path
        self._trace_id = uuid.uuid4().hex[:12]
        self._events: List[TraceEvent] = []
        self._current_span: str = ""

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def event(self, name: str, level: EventLevel = EventLevel.INFO,
              **attributes: Any) -> None:
        """Record a point event (no duration)."""
        if not self.enabled:
            return
        evt = TraceEvent(
            name=name,
            level=level,
            attributes=attributes,
            trace_id=self._trace_id,
            parent_span_id=self._current_span,
        )
        self._record(evt)

    def span(self, name: str, **attributes: Any) -> Span:
        """Create a timed span (use as context manager)."""
        return Span(
            self, name,
            parent_span_id=self._current_span,
            **attributes,
        )

    def llm_call(self, agent: str, model: str, tokens: int = 0,
                 latency_ms: float = 0, **extra: Any) -> None:
        """Record an LLM API call."""
        self.event(
            "llm_call",
            agent=agent, model=model, tokens=tokens,
            latency_ms=latency_ms, **extra,
        )

    def tool_call(self, tool_name: str, input_data: Any = None,
                  output: Any = None, **extra: Any) -> None:
        """Record a tool execution."""
        self.event(
            "tool_call",
            tool_name=tool_name, input=str(input_data)[:200],
            output=str(output)[:200], **extra,
        )

    def state_change(self, key: str, old_value: Any, new_value: Any,
                     **extra: Any) -> None:
        """Record a state change."""
        self.event(
            "state_change",
            level=EventLevel.DEBUG,
            key=key,
            old_value=str(old_value)[:100],
            new_value=str(new_value)[:100],
            **extra,
        )

    def warning(self, message: str, **extra: Any) -> None:
        self.event("warning", level=EventLevel.WARNING, message=message, **extra)

    def error(self, message: str, **extra: Any) -> None:
        self.event("error", level=EventLevel.ERROR, message=message, **extra)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _record(self, evt: TraceEvent) -> None:
        self._events.append(evt)
        if self.console_output:
            self._print_event(evt)

    def _print_event(self, evt: TraceEvent) -> None:
        level_icons = {
            EventLevel.DEBUG: "🔍",
            EventLevel.INFO: "ℹ️",
            EventLevel.WARNING: "⚠️",
            EventLevel.ERROR: "❌",
        }
        icon = level_icons.get(evt.level, "")
        dur = f" ({evt.duration_ms:.0f}ms)" if evt.duration_ms > 0 else ""
        attrs = ", ".join(f"{k}={v}" for k, v in list(evt.attributes.items())[:3])
        _trace_logger.info("%s [%s]%s %s", icon, evt.name, dur, attrs)

    # ------------------------------------------------------------------
    # Export / Analysis
    # ------------------------------------------------------------------

    def export_json(self, path: Optional[str] = None) -> str:
        """Export trace as JSON."""
        path = path or self.export_path
        data = {
            "trace_id": self._trace_id,
            "event_count": len(self._events),
            "events": [e.to_dict() for e in self._events],
        }
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    def summary(self) -> Dict[str, Any]:
        """Generate trace summary statistics."""
        if not self._events:
            return {"total_events": 0}

        llm_events = [e for e in self._events if e.name == "llm_call"]
        tool_events = [e for e in self._events if e.name == "tool_call"]
        errors = [e for e in self._events if e.level == EventLevel.ERROR]

        total_tokens = sum(
            e.attributes.get("tokens", 0) for e in llm_events
        )
        total_latency = sum(
            e.attributes.get("latency_ms", 0) for e in llm_events
        )

        return {
            "trace_id": self._trace_id,
            "total_events": len(self._events),
            "llm_calls": len(llm_events),
            "tool_calls": len(tool_events),
            "errors": len(errors),
            "total_tokens": total_tokens,
            "total_llm_latency_ms": total_latency,
            "avg_llm_latency_ms": total_latency / max(len(llm_events), 1),
        }

    def reset(self) -> None:
        self._events.clear()
        self._trace_id = uuid.uuid4().hex[:12]

    def close(self) -> None:
        """Finalize and optionally export."""
        if self.export_path:
            self.export_json(self.export_path)

    def __len__(self) -> int:
        return len(self._events)
   
    def __bool__(self) -> bool:
        """Tracer is always truthy when it exists (regardless of event count)."""
        return True

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any, Callable


class AgentMemory:
    """Thread-safe memory store keyed by agent id."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._data: dict[str, dict[str, Any]] = {}

    def put(self, agent_id: str, key: str, value: Any) -> None:
        with self._lock:
            self._data.setdefault(agent_id, {})[key] = value

    def get(self, agent_id: str, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(agent_id, {}).get(key, default)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {agent: values.copy() for agent, values in self._data.items()}


class AgentTracer:
    """Simple in-memory trace collector for agent execution events."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._events: list[dict[str, Any]] = []

    def record(self, event_type: str, agent_id: str, **fields: Any) -> None:
        event = {"event": event_type, "agent_id": agent_id, **fields}
        with self._lock:
            self._events.append(event)

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)


class AgentExecutionError(RuntimeError):
    """Raised when one or more agents fail during parallel execution."""

    def __init__(self, results: dict[str, Any], errors: dict[str, Exception]) -> None:
        self.results = results
        self.errors = errors
        super().__init__(f"{len(errors)} agent(s) failed: {', '.join(sorted(errors))}")


@dataclass(frozen=True)
class AgentContext:
    agent_id: str
    memory: AgentMemory
    tracer: AgentTracer


class ParallelAgentExecutor:
    """Run named agent callables in parallel with tracing hooks."""

    def __init__(self, memory: AgentMemory | None = None, tracer: AgentTracer | None = None) -> None:
        self.memory = memory or AgentMemory()
        self.tracer = tracer or AgentTracer()

    def run(self, tasks: dict[str, Callable[[AgentContext], Any]]) -> dict[str, Any]:
        if not tasks:
            raise ValueError("tasks must not be empty")
        results: dict[str, Any] = {agent_id: None for agent_id in tasks}
        errors: dict[str, Exception] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as executor:
            futures = {executor.submit(self._run_one, agent_id, task): agent_id for agent_id, task in tasks.items()}
            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    results[agent_id] = future.result()
                except Exception as exc:
                    errors[agent_id] = exc
        if errors:
            raise AgentExecutionError(results=results, errors=errors)
        return results

    def _run_one(self, agent_id: str, task: Callable[[AgentContext], Any]) -> Any:
        context = AgentContext(agent_id=agent_id, memory=self.memory, tracer=self.tracer)
        self.tracer.record("agent.start", agent_id)
        started = perf_counter()
        try:
            result = task(context)
            self.tracer.record("agent.success", agent_id, duration_ms=(perf_counter() - started) * 1000)
            return result
        except Exception as exc:
            self.tracer.record("agent.error", agent_id, duration_ms=(perf_counter() - started) * 1000, error=str(exc))
            raise

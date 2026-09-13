from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping


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
            return [event.copy() for event in self._events]


class AgentExecutionError(RuntimeError):
    """Raised when one or more agents fail.

    Attributes:
        results: Per-agent output values collected before completion.
        errors: Per-agent raised exceptions for failed agents.
    """

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
    """Run named agent callables in parallel with tracing hooks.

    `run()` raises AgentExecutionError if any agent fails, including partial
    `results` and per-agent `errors`.
    """

    def __init__(
        self,
        memory: AgentMemory | None = None,
        tracer: AgentTracer | None = None,
        max_workers: int | None = None,
    ) -> None:
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be >= 1 when provided")
        self.memory = memory or AgentMemory()
        self.tracer = tracer or AgentTracer()
        self.max_workers = max_workers

    def run(
        self, tasks: Mapping[str, Callable[[AgentContext], Any]] | Iterable[tuple[str, Callable[[AgentContext], Any]]]
    ) -> dict[str, Any]:
        task_items = self._normalize_tasks(tasks)
        if not task_items:
            raise ValueError("tasks must not be empty")
        results: dict[str, Any] = {agent_id: None for agent_id, _ in task_items}
        errors: dict[str, Exception] = {}
        requested_workers = len(task_items) if self.max_workers is None else self.max_workers
        max_workers = min(requested_workers, len(task_items))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._run_one, agent_id, task): agent_id for agent_id, task in task_items}
            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    results[agent_id] = future.result()
                except Exception as exc:
                    errors[agent_id] = exc
        if errors:
            raise AgentExecutionError(results=results, errors=errors)
        return results

    def _normalize_tasks(
        self, tasks: Mapping[str, Callable[[AgentContext], Any]] | Iterable[tuple[str, Callable[[AgentContext], Any]]]
    ) -> list[tuple[str, Callable[[AgentContext], Any]]]:
        task_items = list(tasks.items()) if isinstance(tasks, Mapping) else list(tasks)
        seen_ids: set[str] = set()
        for agent_id, _ in task_items:
            if agent_id in seen_ids:
                raise ValueError(f"duplicate agent_id: {agent_id}")
            seen_ids.add(agent_id)
        return task_items

    def _run_one(self, agent_id: str, task: Callable[[AgentContext], Any]) -> Any:
        context = AgentContext(agent_id=agent_id, memory=self.memory, tracer=self.tracer)
        self.tracer.record("agent.start", agent_id)
        started = perf_counter()
        try:
            result = task(context)
            self.tracer.record("agent.success", agent_id, duration_ms=(perf_counter() - started) * 1000)
            return result
        except Exception as exc:
            self.tracer.record(
                "agent.error",
                agent_id,
                duration_ms=(perf_counter() - started) * 1000,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise

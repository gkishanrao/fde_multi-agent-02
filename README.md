# fde_multi-agent-02

Minimal reference implementation for:
- Agent Memory
- Parallel Execution
- Agent Tracing

## Files

- `agent_runtime.py`
  - `AgentMemory`: thread-safe shared memory per agent
  - `AgentTracer`: in-memory trace event collector
  - `ParallelAgentExecutor`: runs agent callables concurrently with tracing
- `test_agent_runtime.py`
  - Focused unit tests for memory, parallel execution, and tracing

## Run tests

```bash
python -m unittest -v test_agent_runtime.py
```
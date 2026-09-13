# fde_multi-agent-02

Minimal reference implementation for:
- Agent Memory
- Parallel Execution
- Agent Tracing

## Files

- `/home/runner/work/fde_multi-agent-02/fde_multi-agent-02/agent_runtime.py`
  - `AgentMemory`: thread-safe shared memory per agent
  - `AgentTracer`: in-memory trace event collector
  - `ParallelAgentExecutor`: runs agent callables concurrently with tracing
- `/home/runner/work/fde_multi-agent-02/fde_multi-agent-02/test_agent_runtime.py`
  - Focused unit tests for memory, parallel execution, and tracing

## Run tests

```bash
python -m unittest -v /home/runner/work/fde_multi-agent-02/fde_multi-agent-02/test_agent_runtime.py
```
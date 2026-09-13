import threading
import time
import unittest

from agent_runtime import AgentMemory, AgentTracer, ParallelAgentExecutor


class AgentRuntimeTests(unittest.TestCase):
    def test_agent_memory_put_get_snapshot(self) -> None:
        memory = AgentMemory()
        memory.put("agent-a", "topic", "math")
        memory.put("agent-b", "topic", "science")

        self.assertEqual(memory.get("agent-a", "topic"), "math")
        self.assertEqual(memory.get("agent-b", "topic"), "science")
        self.assertEqual(
            memory.snapshot(),
            {
                "agent-a": {"topic": "math"},
                "agent-b": {"topic": "science"},
            },
        )

    def test_parallel_execution_collects_results(self) -> None:
        executor = ParallelAgentExecutor()
        start_barrier = threading.Barrier(2)

        def agent_one(ctx):
            start_barrier.wait(timeout=1)
            time.sleep(0.05)
            ctx.memory.put(ctx.agent_id, "value", 1)
            return "done-1"

        def agent_two(ctx):
            start_barrier.wait(timeout=1)
            time.sleep(0.05)
            ctx.memory.put(ctx.agent_id, "value", 2)
            return "done-2"

        results = executor.run({"agent-1": agent_one, "agent-2": agent_two})

        self.assertEqual(results, {"agent-1": "done-1", "agent-2": "done-2"})
        self.assertEqual(executor.memory.get("agent-1", "value"), 1)
        self.assertEqual(executor.memory.get("agent-2", "value"), 2)

    def test_agent_tracing_records_start_and_success(self) -> None:
        tracer = AgentTracer()
        executor = ParallelAgentExecutor(tracer=tracer)

        def agent(ctx):
            return f"ok-{ctx.agent_id}"

        executor.run({"agent-x": agent})
        events = tracer.events()

        self.assertEqual(events[0]["event"], "agent.start")
        self.assertEqual(events[0]["agent_id"], "agent-x")
        self.assertEqual(events[1]["event"], "agent.success")
        self.assertEqual(events[1]["agent_id"], "agent-x")
        self.assertIn("duration_ms", events[1])


if __name__ == "__main__":
    unittest.main()

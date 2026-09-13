import threading
import time
import unittest

from agent_runtime import AgentExecutionError, AgentMemory, AgentTracer, ParallelAgentExecutor


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
        tracer = AgentTracer()
        executor = ParallelAgentExecutor(tracer=tracer)
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
        start_events = {event["agent_id"] for event in tracer.events() if event["event"] == "agent.start"}
        success_events = {event["agent_id"] for event in tracer.events() if event["event"] == "agent.success"}
        self.assertEqual(start_events, {"agent-1", "agent-2"})
        self.assertEqual(success_events, {"agent-1", "agent-2"})

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

    def test_failing_agent_raises_and_records_error_trace(self) -> None:
        tracer = AgentTracer()
        executor = ParallelAgentExecutor(tracer=tracer)

        def good_agent(_ctx):
            return "ok"

        def bad_agent(_ctx):
            raise ValueError("boom")

        with self.assertRaises(AgentExecutionError) as raised:
            executor.run({"good-agent": good_agent, "bad-agent": bad_agent})

        self.assertEqual(raised.exception.results["good-agent"], "ok")
        self.assertIsNone(raised.exception.results["bad-agent"])
        self.assertIn("bad-agent", raised.exception.errors)

        error_events = [event for event in tracer.events() if event["event"] == "agent.error"]
        self.assertEqual(len(error_events), 1)
        self.assertEqual(error_events[0]["agent_id"], "bad-agent")
        self.assertEqual(error_events[0]["error"], "boom")
        self.assertIn("duration_ms", error_events[0])

    def test_empty_task_set_is_rejected(self) -> None:
        executor = ParallelAgentExecutor()
        with self.assertRaisesRegex(ValueError, "tasks must not be empty"):
            executor.run({})


if __name__ == "__main__":
    unittest.main()

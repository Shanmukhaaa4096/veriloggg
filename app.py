"""
Verilog Code Generator and Tester Agent.
Deployment requirement: iverilog and vvp (Icarus Verilog) must be on PATH.
"""
import os
import re
import subprocess
import tempfile
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langserve import add_routes

MAX_ITERATIONS = 5

# kept original API-key env var. Model name unverifiable against a live
# model list here -> swap if your account rejects it.
llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
)


class AgentState(TypedDict):
    spec: str
    code: Optional[str]
    testbench: Optional[str]
    iteration: int
    max_iterations: int
    last_error: Optional[str]
    critique_history: List[str]
    verified: bool
    log: List[str]  # per-iteration (attempt, pass/fail, summary)


def _extract_block(text: str, tag: str) -> str:
    m = re.search(rf"```{tag}\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(rf"{tag.upper()}:(.*?)(?:{'TESTBENCH' if tag=='verilog' else 'MODULE'}:|$)",
                  text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def generator(state: AgentState) -> dict:
    """Writes/revises Verilog module + testbench from spec and prior critique."""
    critique = state["critique_history"][-1] if state["critique_history"] else None
    prompt = (
        "You are a Verilog engineer. Write a synthesizable Verilog module and a "
        "self-checking testbench for this spec:\n"
        f"{state['spec']}\n\n"
        "Return exactly two fenced code blocks, first ```verilog ... ``` for the "
        "module, second ```testbench ... ``` for the testbench. The testbench must "
        "print 'TEST PASSED' on success and 'TEST FAILED' plus a reason on any "
        "mismatch, then $finish."
    )
    if critique:
        prompt += f"\n\nPrevious attempt failed. Critique to fix:\n{critique}"
        if state.get("code"):
            prompt += f"\n\nPrevious module:\n{state['code']}"
        if state.get("testbench"):
            prompt += f"\n\nPrevious testbench:\n{state['testbench']}"

    resp = llm.invoke([HumanMessage(content=prompt)])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)

    return {
        "code": _extract_block(text, "verilog"),
        "testbench": _extract_block(text, "testbench"),
        "iteration": state["iteration"] + 1,
    }


def simulator(state: AgentState) -> dict:
    """Compiles + runs code+testbench with iverilog/vvp. No LLM guessing."""
    with tempfile.TemporaryDirectory() as d:
        design = os.path.join(d, "design.v")
        tb = os.path.join(d, "tb.v")
        out = os.path.join(d, "sim.out")
        with open(design, "w") as f:
            f.write(state["code"])
        with open(tb, "w") as f:
            f.write(state["testbench"])

        compile_proc = subprocess.run(
            ["iverilog", "-o", out, design, tb],
            capture_output=True, text=True, timeout=30,
        )
        if compile_proc.returncode != 0:
            return {"last_error": f"COMPILE ERROR:\n{compile_proc.stderr}", "verified": False}

        run_proc = subprocess.run(
            ["vvp", out], capture_output=True, text=True, timeout=30,
        )
        sim_out = run_proc.stdout + run_proc.stderr
        passed = "TEST PASSED" in sim_out and "TEST FAILED" not in sim_out
        return {
            "last_error": None if passed else f"SIMULATION OUTPUT:\n{sim_out}",
            "verified": passed,
        }


def critic(state: AgentState) -> dict:
    """LLM reviews raw simulator failure output, writes an actionable critique."""
    prompt = (
        "You are a strict Verilog code reviewer. The following attempt failed. "
        "Point out exactly what is wrong (syntax, port mismatch, failed vectors, "
        "logic bug) and how to fix it. Be specific, no vague praise, no insults "
        "without a fix attached.\n\n"
        f"Spec:\n{state['spec']}\n\nModule:\n{state['code']}\n\n"
        f"Testbench:\n{state['testbench']}\n\nError/output:\n{state['last_error']}"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {"critique_history": state["critique_history"] + [text]}


def log_iteration(state: AgentState) -> dict:
    status = "PASS" if state["verified"] else "FAIL"
    summary = "verified" if state["verified"] else (state["last_error"] or "")[:200]
    entry = f"attempt {state['iteration']}: {status} - {summary}"
    return {"log": state["log"] + [entry]}


def route_after_sim(state: AgentState) -> str:
    if state["verified"]:
        return "log_pass"
    if state["iteration"] >= state["max_iterations"]:
        return "log_fail_stop"
    return "log_fail_retry"


graph = StateGraph(AgentState)
graph.add_node("generator", generator)
graph.add_node("simulator", simulator)
graph.add_node("critic", critic)
graph.add_node("log_pass", log_iteration)
graph.add_node("log_fail_retry", log_iteration)
graph.add_node("log_fail_stop", log_iteration)

graph.add_edge(START, "generator")
graph.add_edge("generator", "simulator")
graph.add_conditional_edges("simulator", route_after_sim, {
    "log_pass": "log_pass",
    "log_fail_retry": "log_fail_retry",
    "log_fail_stop": "log_fail_stop",
})
graph.add_edge("log_pass", END)
graph.add_edge("log_fail_retry", "critic")
graph.add_edge("critic", "generator")
graph.add_edge("log_fail_stop", END)

compiled_graph = graph.compile()


class VerilogSpecInput(BaseModel):
    spec: str


def to_state(inp) -> AgentState:
    spec = inp["spec"] if isinstance(inp, dict) else inp.spec
    return {
        "spec": spec,
        "code": None,
        "testbench": None,
        "iteration": 0,
        "max_iterations": MAX_ITERATIONS,
        "last_error": None,
        "critique_history": [],
        "verified": False,
        "log": [],
    }


runnable = to_state | compiled_graph  # RunnableSequence: pydantic input -> AgentState -> graph

app = FastAPI(title="Verilog Code Generator and Tester")
add_routes(app, runnable, path="/agent", input_type=VerilogSpecInput)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import re
import subprocess
import tempfile
import uvicorn
from typing import TypedDict, List, Optional
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# ==========================================
# DEPLOYMENT REQUIREMENT
# Icarus Verilog (iverilog, vvp) must be installed and on PATH.
# Debian/Ubuntu: apt-get install iverilog
# ==========================================

MAX_ITERATIONS = 5

# --- 1. LLM ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
    temperature=0,
)

# --- 2. STATE ---
class VerilogState(TypedDict):
    spec: str                # natural-language hardware spec
    code: Optional[str]      # current Verilog module
    testbench: Optional[str] # current testbench
    iteration: int           # attempts so far
    passed: bool             # simulator success flag
    last_error: Optional[str]      # raw simulator output from last attempt
    critique_history: List[str]    # critic notes per iteration


# --- 3. HELPERS ---
def _strip_code_fence(text: str) -> str:
    """Remove ```verilog / ``` fences the LLM sometimes adds."""
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def _extract_module_name(code: str, default: str) -> str:
    m = re.search(r"\bmodule\s+(\w+)", code or "")
    return m.group(1) if m else default


# --- 4. GRAPH NODES ---
def generator_node(state: VerilogState) -> dict:
    """Writes (or revises, using the last critique) the Verilog design + testbench."""
    if state["iteration"] == 0:
        prompt = (
            "You are a Verilog RTL engineer. Write synthesizable Verilog for this spec:\n"
            f"{state['spec']}\n\n"
            "Then write a self-checking testbench that instantiates the module, applies "
            "stimulus, checks results, and prints exactly 'TEST PASSED' if all checks pass "
            "or 'TEST FAILED' if any check fails, then calls $finish.\n"
            "Return two fenced code blocks in this exact order and nothing else:\n"
            "```verilog\n<design module>\n```\n```verilog\n<testbench module named tb>\n```"
        )
    else:
        prompt = (
            "You are a Verilog RTL engineer fixing a design after a failed simulation.\n"
            f"Spec:\n{state['spec']}\n\n"
            f"Previous design:\n{state['code']}\n\n"
            f"Previous testbench:\n{state['testbench']}\n\n"
            f"Critique of what's wrong:\n{state['critique_history'][-1]}\n\n"
            "Fix the design and/or testbench. Return two fenced code blocks in this exact "
            "order and nothing else:\n"
            "```verilog\n<design module>\n```\n```verilog\n<testbench module named tb>\n```"
        )

    response = llm.invoke(prompt)
    text = response.content if isinstance(response.content, str) else str(response.content)

    blocks = re.findall(r"```(?:verilog)?\n(.*?)```", text, re.DOTALL)
    if len(blocks) >= 2:
        design, testbench = blocks[0].strip(), blocks[1].strip()
    else:
        # Model didn't follow format -> keep whole reply as design, reuse old testbench.
        design = _strip_code_fence(text)
        testbench = state.get("testbench") or ""

    return {"code": design, "testbench": testbench, "iteration": state["iteration"] + 1}


def simulator_node(state: VerilogState) -> dict:
    """Compiles + runs the design and testbench with Icarus Verilog. Real execution,
    not an LLM guess: iverilog compiles, vvp simulates, we read the actual output."""
    design_name = _extract_module_name(state["code"], "design")
    tb_name = _extract_module_name(state["testbench"], "tb")

    with tempfile.TemporaryDirectory() as tmp:
        design_path = os.path.join(tmp, f"{design_name}.v")
        tb_path = os.path.join(tmp, f"{tb_name}.v")
        out_path = os.path.join(tmp, "sim.out")

        with open(design_path, "w") as f:
            f.write(state["code"] or "")
        with open(tb_path, "w") as f:
            f.write(state["testbench"] or "")

        compile_proc = subprocess.run(
            ["iverilog", "-o", out_path, design_path, tb_path],
            capture_output=True, text=True, timeout=30,
        )
        if compile_proc.returncode != 0:
            return {"passed": False, "last_error": f"COMPILE ERROR:\n{compile_proc.stderr}"}

        run_proc = subprocess.run(
            ["vvp", out_path], capture_output=True, text=True, timeout=30,
        )
        output = run_proc.stdout + run_proc.stderr

        if run_proc.returncode != 0:
            return {"passed": False, "last_error": f"SIMULATION ERROR:\n{output}"}
        if "TEST FAILED" in output or "TEST PASSED" not in output:
            return {"passed": False, "last_error": f"TESTBENCH REPORT:\n{output}"}

        return {"passed": True, "last_error": output}


def critic_node(state: VerilogState) -> dict:
    """LLM reviews the raw simulator output and writes a concrete, actionable critique."""
    prompt = (
        "You are a strict Verilog reviewer. The simulation below failed. Point out exactly "
        "what's wrong (syntax errors, port mismatches, failed test vectors, timing bugs) and "
        "what must change to fix it. Be blunt and specific, no vague comments.\n\n"
        f"Design:\n{state['code']}\n\nTestbench:\n{state['testbench']}\n\n"
        f"Simulator output:\n{state['last_error']}"
    )
    response = llm.invoke(prompt)
    critique = response.content if isinstance(response.content, str) else str(response.content)
    return {"critique_history": state["critique_history"] + [critique]}


# --- 5. ROUTER ---
def route_after_simulation(state: VerilogState) -> str:
    if state["passed"]:
        return "end"
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "critic"


# --- 6. GRAPH CONSTRUCTION ---
workflow = StateGraph(VerilogState)
workflow.add_node("generator", generator_node)
workflow.add_node("simulator", simulator_node)
workflow.add_node("critic", critic_node)

workflow.add_edge(START, "generator")
workflow.add_edge("generator", "simulator")
workflow.add_conditional_edges("simulator", route_after_simulation, {"critic": "critic", "end": END})
workflow.add_edge("critic", "generator")

verilog_app = workflow.compile()


# --- 7. LANGSERVE WRAPPER ---
class VerilogInput(BaseModel):
    spec: str = Field(description="Natural-language hardware spec, e.g. '4-bit synchronous up-counter with async reset'")


def _init_state(x) -> dict:
    spec = x["spec"] if isinstance(x, dict) else x.spec
    return {
        "spec": spec, "code": None, "testbench": None, "iteration": 0,
        "passed": False, "last_error": None, "critique_history": [],
    }


def _format_output(state: dict) -> dict:
    return {
        "code": state.get("code"),
        "testbench": state.get("testbench"),
        "verified": state.get("passed", False),
        "iterations": state.get("iteration", 0),
        "last_error": state.get("last_error"),
        "critique_history": state.get("critique_history", []),
    }


verilog_chain = (
    RunnableLambda(_init_state)
    | verilog_app
    | RunnableLambda(_format_output)
).with_types(input_type=VerilogInput, output_type=dict)

# --- 8. FASTAPI APP ---
app = FastAPI()
add_routes(app, verilog_chain, path="/agent")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

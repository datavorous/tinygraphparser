"""Runtime partition analysis via apply_plugin_main + rewritten flatbuffer.

apply_plugin_main runs the QNN compiler plugin and rewrites the .tflite with
DISPATCH_OP nodes. This module:
  - Invokes apply_plugin_main and captures logs + rewritten flatbuffer
  - Parses the rewritten flatbuffer: DISPATCH_OP presence = delegated
  - Parses the log for ValidateOp rejection reasons (error code + op type)
  - Merges into RuntimeResult per subgraph

Log format sources (verified against source):
  Rejection  - qnn_manager.cc:411:
    LITERT_LOG(LITERT_ERROR, "Failed to validate op %s\\n, error: %lld", name, error)
    Line 1: "ERROR: [qnn_manager.cc:412] Failed to validate op <name>"
    Line 2: ", error: <code>"
    Note: \\n literal in format string splits the message.

  Partition summary - compiler_plugin.cc:702:
    LITERT_LOG(LITERT_INFO,
      "Partitioned subgraph<%d>, selected %lu ops, from a total of %lu ops. resulted in %lu partitions.", ...)
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .graph_parser import RED, RESET

# ---------------------------------------------------------------------------
# Known QNN error codes
# ---------------------------------------------------------------------------

_ERROR_LABELS = {
    3110: "dtype_mismatch",
}

# ---------------------------------------------------------------------------
# Log regexes (from source, see module docstring)
# ---------------------------------------------------------------------------

_REJECTION_RE = re.compile(r"Failed to validate op ([^\n]+)\n, error: (-?\d+)")

# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------


@dataclass
class RuntimeResult:
    subgraph: str
    total_ops: int
    delegated_op_count: int
    non_delegated_op_count: int
    delegated_op_indices: List[int]
    non_delegated_op_indices: List[int]
    non_delegated_op_names: Counter = field(default_factory=Counter)
    error_code_histogram: Counter = field(default_factory=Counter)
    rejected_op_type_histogram: Counter = field(default_factory=Counter)


# ---------------------------------------------------------------------------
# RuntimeAnalyser
# ---------------------------------------------------------------------------


class RuntimeAnalyser:
    """Run apply_plugin_main, parse results, and report runtime placement."""

    def __init__(
        self,
        apply_plugin_main_path: str,
        plugin_path: str,
        soc: str,
        qnn_lib_dir: str = "",
    ):
        self._binary = Path(apply_plugin_main_path)
        self._plugin = Path(plugin_path)
        self._soc = soc
        self._qnn_lib = qnn_lib_dir
        self._global_error_hist: Counter = Counter()
        self._global_type_hist: Counter = Counter()

    def run(self, model_path: str, output_dir: str) -> tuple[str, str]:
        """
        Invoke apply_plugin_main. Writes rewritten.tflite and run.log to output_dir.
        Returns (rewritten_path, log_path).
        Raises RuntimeError if binary is missing or tool fails to produce a summary.
        """
        if not self._binary.exists():
            raise RuntimeError(f"apply_plugin_main not found: {self._binary}")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rewritten = str(out / "rewritten.tflite")
        log_path = str(out / "run.log")

        env = os.environ.copy()
        if self._qnn_lib:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{self._qnn_lib}:{existing}" if existing else self._qnn_lib
            )

        result = subprocess.run(
            [
                str(self._binary),
                f"--model={model_path}",
                "--soc_manufacturer=Qualcomm",
                f"--soc_model={self._soc}",
                f"--libs={self._plugin.parent}",
                f"--o={rewritten}",
            ],
            capture_output=True,
            timeout=300,
            env=env,
        )
        log_text = (result.stderr + result.stdout).decode("utf-8", errors="replace")
        Path(log_path).write_text(log_text)
        if result.returncode != 0:
            raise RuntimeError(
                f"apply_plugin_main exited with code {result.returncode}.\n"
                f"stderr tail:\n{result.stderr.decode('utf-8', errors='replace')[-2000:]}"
            )
        return rewritten, log_path

    def parse(self, rewritten_path: str, log_path: str) -> List[RuntimeResult]:
        """
        Parse rewritten flatbuffer (delegated/non_delegated per op index) and
        log (error code histogram, rejected op type histogram). Returns one
        RuntimeResult per subgraph.
        """
        log_text = Path(log_path).read_text(errors="replace")
        rejections = _parse_log(log_text)
        fb_results = _parse_rewritten_flatbuffer(rewritten_path)

        # Log rejections are global (no subgraph attribution in source).
        # Store on the analyser; report() prints them once at the end.
        def _op_type(name: str) -> str:
            return re.sub(r"_\d+$", "", name)

        self._global_error_hist = Counter(code for _, code in rejections)
        self._global_type_hist = Counter(_op_type(name) for name, _ in rejections)

        results = []
        for fb in fb_results:
            is_named = not fb["subgraph"].startswith("subgraph_")
            has_dispatch = len(fb["delegated_op_indices"]) > 0
            # Skip anonymous helper subgraphs injected by apply_plugin_main
            # that have no delegated ops — they're internal QNN partitions, not
            # the original model signatures.
            if not is_named and not has_dispatch:
                continue
            results.append(
                RuntimeResult(
                    subgraph=fb["subgraph"],
                    total_ops=fb["total_ops"],
                    delegated_op_count=len(fb["delegated_op_indices"]),
                    non_delegated_op_count=len(fb["non_delegated_ops"]),
                    delegated_op_indices=fb["delegated_op_indices"],
                    non_delegated_op_indices=[
                        op["op_index"] for op in fb["non_delegated_ops"]
                    ],
                    non_delegated_op_names=Counter(
                        op["opname"] for op in fb["non_delegated_ops"]
                    ),
                )
            )
        return results

    def report(self, results: List[RuntimeResult]) -> None:
        """Print runtime placement report."""
        for i, r in enumerate(results):
            if i:
                print()
            _report_runtime_result(r)
        if self._global_error_hist or self._global_type_hist:
            print()
            _report_global_log(self._global_error_hist, self._global_type_hist)


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------


def _parse_log(log: str) -> list[tuple[str, int]]:
    return [
        (m.group(1).strip(), int(m.group(2))) for m in _REJECTION_RE.finditer(log)
    ]


def _parse_rewritten_flatbuffer(path: str) -> list[dict]:
    """
    Walk the rewritten .tflite from apply_plugin_main.
    DISPATCH_OP: builtin_code=0 (CUSTOM) + CustomCode() == b'DISPATCH_OP' → delegated.
    Everything else → non_delegated.

    Note: TFLiteGraphParser maps builtin=0 to ADD for custom ops (wrong).
    Read CustomCode() directly via the flatbuffers API.
    """
    from tflite.Model import Model
    import tflite.BuiltinOperator as BuiltinOperator

    enum_op = {
        getattr(BuiltinOperator, k): k
        for k in dir(BuiltinOperator)
        if k.isupper() and isinstance(getattr(BuiltinOperator, k), int)
    }

    with open(path, "rb") as f:
        model = Model.GetRootAsModel(f.read(), 0)

    results = []
    for si in range(model.SubgraphsLength()):
        sg = model.Subgraphs(si)
        name = sg.Name().decode() if sg.Name() else f"subgraph_{si}"
        delegated, non_delegated = [], []
        for oi in range(sg.OperatorsLength()):
            op = sg.Operators(oi)
            opc = model.OperatorCodes(op.OpcodeIndex())
            if opc.CustomCode() == b"DISPATCH_OP":
                delegated.append(oi)
            else:
                opname = enum_op.get(opc.BuiltinCode(), f"BUILTIN_{opc.BuiltinCode()}")
                non_delegated.append({"op_index": oi, "opname": opname})
        results.append(
            {
                "subgraph": name,
                "total_ops": sg.OperatorsLength(),
                "delegated_op_indices": delegated,
                "non_delegated_ops": non_delegated,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _report_runtime_result(r: RuntimeResult) -> None:
    print(f"Signature: {r.subgraph}")
    print(
        f"  {RED}Non-delegated original ops: {r.non_delegated_op_count}{RESET}"
    )
    if r.non_delegated_op_names:
        print("  Top non-delegated op types:")
        for name, count in r.non_delegated_op_names.most_common(5):
            print(f"    {count:>5}  {name}")


def _report_global_log(error_hist: Counter, type_hist: Counter) -> None:
    if not error_hist and not type_hist:
        return
    print("Global (log-derived)  -- not attributed to any subgraph")
    if error_hist:
        print("  ValidateOp rejection codes:")
        for code, count in error_hist.most_common():
            label = _ERROR_LABELS.get(code, f"unknown_{code}")
            print(f"    {RED}{code}{RESET}  {label:<20}  {count}")
    if type_hist:
        print("  Top rejected op types:")
        for name, count in type_hist.most_common(10):
            print(f"    {count:>5}  {name}")

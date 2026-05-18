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

# Two-line rejection: name on line N, ", error: <code>" on line N+1.
_REJECTION_NAME_RE = re.compile(r"Failed to validate op (.+?)\s*$")
_REJECTION_ERR_RE = re.compile(r"^,\s*error:\s*(-?\d+)")

# Partition summary (one line, compiler_plugin.cc:702).
_SUMMARY_RE = re.compile(
    r"Partitioned subgraph<(\d+)>, selected (\d+) ops, "
    r"from a total of (\d+) ops\. resulted in (\d+) partitions\."
)

# Dtype-mismatch block (HTP backend ValidateOpConfig).
_UNSUP_RE = re.compile(
    r"Unsupported input/output datatypes requested for the HTP Op '([^']+)' "
    r"in the node '([^']+)'"
)
_REQ_HDR = "Requested I/O datatype set:"
_SUP_HDR_RE = re.compile(r"Supported I/O datatype sets for the configuration:\s*(\S+)")
_DTYPE_LINE_RE = re.compile(r"\b(in|out)\[\d+\]:QNN_DATATYPE_([A-Z_0-9]+)")

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
        self._partition_summaries: List[dict] = []
        self._dtype_rejection_patterns: List[dict] = []

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
        log (rejections, partition summaries, dtype patterns). Returns one
        RuntimeResult per subgraph; per-log aggregates live on the analyser.
        """
        rejections, summaries, patterns = _parse_log_streaming(log_path)
        fb_results = _parse_rewritten_flatbuffer(rewritten_path)

        # Log rejections are global (no subgraph attribution in source).
        # Store on the analyser; report() prints them once at the end.
        def _op_type(name: str) -> str:
            return re.sub(r"_\d+$", "", name)

        self._global_error_hist = Counter(code for _, code in rejections)
        self._global_type_hist = Counter(_op_type(name) for name, _ in rejections)
        self._partition_summaries = summaries
        self._dtype_rejection_patterns = patterns

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
        has_global = (
            self._global_error_hist
            or self._global_type_hist
            or self._partition_summaries
            or self._dtype_rejection_patterns
        )
        if has_global:
            print()
            _report_global_log(
                self._global_error_hist,
                self._global_type_hist,
                self._partition_summaries,
                self._dtype_rejection_patterns,
            )


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------


def _parse_log_streaming(
    log_path: str,
) -> tuple[list[tuple[str, int]], list[dict], list[dict]]:
    """
    Single line-by-line pass over the log. Extracts:
      - rejections: list of (op_name, error_code) tuples
      - partition_summaries: per-subgraph-index ops counts
      - dtype patterns: deduplicated (op_type, requested, supported) tuples

    Streams the file — does NOT load it into memory.
    """
    rejections: list[tuple[str, int]] = []
    summaries: list[dict] = []
    patterns_by_key: dict[tuple, dict] = {}

    pending_op: str | None = None
    cur: dict | None = None
    mode: str | None = None

    def _flush(block: dict | None) -> None:
        if block is None or not block["requested"]:
            return
        key = (
            block["op_type"],
            tuple(block["requested"]),
            tuple(block["supported"]),
        )
        if key not in patterns_by_key:
            patterns_by_key[key] = {
                "op_type": block["op_type"],
                "node_name": block["node_name"],
                "requested": ", ".join(block["requested"]),
                "supported": list(block["supported"]),
            }

    with open(log_path, "r", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # --- rejection two-liner: error code follows the name ---
            if pending_op is not None:
                m = _REJECTION_ERR_RE.match(line.lstrip())
                if m:
                    rejections.append((pending_op, int(m.group(1))))
                pending_op = None

            m = _REJECTION_NAME_RE.search(line)
            if m:
                pending_op = m.group(1).strip()

            # --- partition summary ---
            m = _SUMMARY_RE.search(line)
            if m:
                summaries.append(
                    {
                        "subgraph_index": int(m.group(1)),
                        "selected_ops": int(m.group(2)),
                        "total_ops": int(m.group(3)),
                        "partitions": int(m.group(4)),
                    }
                )
                continue

            # --- dtype-mismatch block ---
            m = _UNSUP_RE.search(line)
            if m:
                _flush(cur)
                cur = {
                    "op_type": m.group(1),
                    "node_name": m.group(2),
                    "requested": [],
                    "supported": [],
                }
                mode = None
                continue

            if cur is None:
                continue

            if _REQ_HDR in line:
                mode = "requested"
                continue

            m = _SUP_HDR_RE.search(line)
            if m:
                mode = "supported"
                label = m.group(1)
                if label not in cur["supported"]:
                    cur["supported"].append(label)
                continue

            if mode == "requested":
                for slot, dt in _DTYPE_LINE_RE.findall(line):
                    token = f"{slot}:{dt}"
                    if token not in cur["requested"]:
                        cur["requested"].append(token)

    _flush(cur)
    return rejections, summaries, list(patterns_by_key.values())


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


def _report_global_log(
    error_hist: Counter,
    type_hist: Counter,
    summaries: list[dict],
    patterns: list[dict],
) -> None:
    print("Global (log-derived)  -- not attributed to any subgraph")
    if summaries:
        print("  Partition summaries (log-derived):")
        for s in summaries:
            print(
                f"    subgraph {s['subgraph_index']}: {s['selected_ops']}/"
                f"{s['total_ops']} ops selected, {s['partitions']} partitions"
            )
    if error_hist:
        print("  ValidateOp rejection codes:")
        for code, count in error_hist.most_common():
            label = _ERROR_LABELS.get(code, f"unknown_{code}")
            print(f"    {RED}{code}{RESET}  {label:<20}  {count}")
    if type_hist:
        print("  Top rejected op types:")
        for name, count in type_hist.most_common(10):
            print(f"    {count:>5}  {name}")
    if patterns:
        print("  Dtype rejection patterns (deduplicated):")
        for p in patterns:
            print(
                f"    {p['op_type']}  requested={p['requested']}  "
                f"supported={p['supported']}"
            )

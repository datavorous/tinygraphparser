"""Static fragmentation analysis for the LiteRT QNN delegate.

Checks two statically visible blockers per op:
  - opSupportMap.csv      : whether a QNN builder exists for the op
  - find_dynamic_shape_ops: whether shape/index inputs are runtime tensors

Does NOT check dtype constraints, attribute constraints, or
backendValidateOpConfig. Real runtime partitioning may differ.

A partition is a maximal contiguous run of ops with the same eligibility.
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .graph_parser import find_dynamic_shape_ops, GREEN, RED, RESET


# ---------------------------------------------------------------------------
# opSupportMap.csv parser
# ---------------------------------------------------------------------------

# Irregular spellings the naive CamelCase->SNAKE converter gets wrong.
_NAME_OVERRIDES: Dict[str, str] = {
    "ReluN1To1":       "RELU_N1_TO_1",
    "TopkV2":          "TOPK_V2",
    "Padv2":           "PADV2",
    "ReverseV2":       "REVERSE_V2",
    "SelectV2":        "SELECT_V2",
    "L2Pool2d":        "L2_POOL_2D",
    "L2Normalization": "L2_NORMALIZATION",
    "BatchMatmul":     "BATCH_MATMUL",
}

_TFL_PREFIX = "kLiteRtOpCodeTfl"
_SHLO_PREFIX = "kLiteRtOpCodeShloComposite"


def _camel_to_upper_snake(camel: str) -> str:
    if camel in _NAME_OVERRIDES:
        return _NAME_OVERRIDES[camel]
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1_\2", s)
    s = re.sub(r"(\d)([A-Z])", r"\1_\2", s)
    return s.upper()


@dataclass
class OpSupport:
    tfl_supported: Set[str] = field(default_factory=set)
    builder_file: Dict[str, str] = field(default_factory=dict)
    composite_supported: Set[str] = field(default_factory=set)
    rows: List[Dict[str, str]] = field(default_factory=list)


def load_op_support(csv_path: str) -> OpSupport:
    """Parse the tab-separated opSupportMap.csv into an OpSupport struct."""
    out = OpSupport()
    with open(csv_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out.rows.append(row)
            code = row["litert_op_code"]

            if code.startswith(_TFL_PREFIX):
                name = _camel_to_upper_snake(code[len(_TFL_PREFIX):])
                out.tfl_supported.add(name)
                out.builder_file[name] = row.get("builder_file", "")

            elif code.startswith(_SHLO_PREFIX):
                rest = code[len(_SHLO_PREFIX):]
                if rest.startswith(":"):
                    m = re.match(r":(k[A-Za-z0-9]+)", rest)
                    if m:
                        out.composite_supported.add(m.group(1))
                else:
                    # composite_supported per-name is not consulted during
                    # classification; composite handling here is coarser than
                    # real QNN dispatch.
                    out.tfl_supported.add("STABLEHLO_COMPOSITE")
                    out.tfl_supported.add("SHLO_COMPOSITE")
    return out


# ---------------------------------------------------------------------------
# Partition simulator
# ---------------------------------------------------------------------------

@dataclass
class Partition:
    kind: str                       # 'NPU' | 'CPU'
    op_indices: Tuple[int, int]     # inclusive (start, end) in subgraph order
    op_count: int
    subgraph: str
    reason: Optional[str] = None    # CPU only: 'no_builder' | 'dynamic_shape' | 'unsupported_composite'
    op_breakdown: Counter = field(default_factory=Counter)


@dataclass
class PartitionResult:
    subgraph: str
    total_ops: int
    partitions: List[Partition]


_COMPOSITE_NAMES = ("STABLEHLO_COMPOSITE", "SHLO_COMPOSITE")


def _classify_op(op: Dict[str, Any], op_support: OpSupport, dynamic_indices: Set[int]) -> Tuple[bool, Optional[str]]:
    # Only checks builder presence and dynamic shape; dtype and attribute constraints are not modeled.
    if op["index"] in dynamic_indices:
        return False, "dynamic_shape"

    opname = op["opname"]
    if opname in _COMPOSITE_NAMES:
        if any(c in op_support.tfl_supported for c in _COMPOSITE_NAMES):
            return True, None
        return False, "unsupported_composite"

    if opname in op_support.tfl_supported:
        return True, None
    return False, "no_builder"


def simulate_partition(graph: Dict[str, Any], op_support: OpSupport) -> List[PartitionResult]:
    """Classify ops by static eligibility. Returns one PartitionResult per subgraph."""
    dyn_by_sg: Dict[str, Set[int]] = defaultdict(set)
    for d in find_dynamic_shape_ops(graph):
        dyn_by_sg[d["subgraph"]].add(d["op_index"])

    results: List[PartitionResult] = []
    for sg in graph["subgraphs"]:
        ops = sg["ops"]
        dyn = dyn_by_sg[sg["name"]]
        partitions = _partition_subgraph(sg["name"], ops, op_support, dyn)
        results.append(PartitionResult(subgraph=sg["name"], total_ops=len(ops), partitions=partitions))
    return results


def _partition_subgraph(sg_name: str, ops: List[Dict[str, Any]],
                         op_support: OpSupport, dynamic_indices: Set[int]) -> List[Partition]:
    partitions: List[Partition] = []
    run_start = 0
    run_eligible: Optional[bool] = None
    run_reason: Optional[str] = None
    run_breakdown: Counter = Counter()

    def flush(end: int):
        nonlocal run_breakdown
        partitions.append(Partition(
            kind="NPU" if run_eligible else "CPU",
            op_indices=(ops[run_start]["index"], ops[end - 1]["index"]),
            op_count=end - run_start,
            subgraph=sg_name,
            reason=None if run_eligible else run_reason,
            op_breakdown=Counter() if run_eligible else run_breakdown,
        ))
        run_breakdown = Counter()

    for i, op in enumerate(ops):
        eligible, reason = _classify_op(op, op_support, dynamic_indices)

        # A CPU run continues only if the reason matches; different reasons
        # become different partitions so the breakdown stays actionable.
        same_run = (run_eligible is not None and eligible == run_eligible
                    and (eligible or reason == run_reason))

        if not same_run:
            if run_eligible is not None:
                flush(i)
            run_start = i
            run_eligible = eligible
            run_reason = reason

        if not eligible:
            run_breakdown[op["opname"]] += 1

    if run_eligible is not None:
        flush(len(ops))
    return partitions


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_partition(result: PartitionResult) -> None:
    parts = result.partitions
    npu = [p for p in parts if p.kind == "NPU"]
    cpu = [p for p in parts if p.kind == "CPU"]

    print(f"Signature: {result.subgraph} ({result.total_ops} ops)")
    print(f"  Partitions: {len(parts)} total - {len(npu)} NPU, {len(cpu)} CPU")

    if npu:
        largest = max(npu, key=lambda p: p.op_count)
        smallest = min(npu, key=lambda p: p.op_count)
        avg = sum(p.op_count for p in npu) / len(npu)
        print(f"  Largest  NPU partition: {GREEN}{largest.op_count}{RESET} ops "
              f"(ops {largest.op_indices[0]}-{largest.op_indices[1]})")
        print(f"  Smallest NPU partition: {RED}{smallest.op_count}{RESET} ops "
              f"(ops {smallest.op_indices[0]}-{smallest.op_indices[1]})")
        print(f"  Mean     NPU partition: {avg:.1f} ops")
    else:
        print(f"  {RED}No NPU partitions{RESET}")

    if cpu:
        by_reason: Dict[str, Counter] = defaultdict(Counter)
        reason_totals: Counter = Counter()
        for p in cpu:
            reason_totals[p.reason] += p.op_count
            by_reason[p.reason].update(p.op_breakdown)
        print("  CPU fallback breakdown:")
        for reason, total in reason_totals.most_common():
            ops_str = ", ".join(f"{n} x{c}" for n, c in by_reason[reason].most_common())
            print(f"    {RED}{reason}{RESET}: {total} ops [{ops_str}]")


def report_partitions(results: List[PartitionResult]) -> None:
    for r in results:
        report_partition(r)
        print()


# ---------------------------------------------------------------------------
# Seam dump - context around partition boundaries
# ---------------------------------------------------------------------------

def _fmt_op_oneline(op: Dict[str, Any]) -> str:
    dtype = op["outputs"][0]["dtype"] if op["outputs"] else (op["inputs"][0]["dtype"] if op["inputs"] else "?")
    in_shape = op["inputs"][0]["shape"] if op["inputs"] else []
    out_shape = op["outputs"][0]["shape"] if op["outputs"] else []
    return f"{op['opname']} {dtype} {in_shape} -> {out_shape}"


def report_seams(graph: Dict[str, Any], results: List[PartitionResult],
                 context: int = 2, kind: str = "CPU") -> None:
    """Print ops surrounding every CPU (or NPU) partition. Useful for diagnosing why a run split."""
    ops_by_sg = {sg["name"]: sg["ops"] for sg in graph["subgraphs"]}

    for r in results:
        ops = ops_by_sg[r.subgraph]
        pos_by_idx = {op["index"]: i for i, op in enumerate(ops)}
        targets = [p for p in r.partitions if p.kind == kind]
        if not targets:
            continue

        print(f"=== Seams in {r.subgraph} ({len(targets)} {kind} partition(s)) ===")
        for i, p in enumerate(r.partitions):
            if p.kind != kind:
                continue
            tag = f"Partition {i} ({p.kind}, {p.op_count} op{'s' if p.op_count != 1 else ''}"
            if p.reason:
                tag += f", reason={p.reason}"
            print(tag + ")")

            start_pos = pos_by_idx[p.op_indices[0]]
            end_pos = pos_by_idx[p.op_indices[1]]

            for j in range(max(0, start_pos - context), start_pos):
                print(f"  prev: op {ops[j]['index']} {_fmt_op_oneline(ops[j])}")

            inside = ops[start_pos:end_pos + 1]
            if len(inside) <= 2 * context + 1:
                for op in inside:
                    print(f"  {RED}>>>{RESET}  op {op['index']} {_fmt_op_oneline(op)}")
            else:
                for op in inside[:context]:
                    print(f"  {RED}>>>{RESET}  op {op['index']} {_fmt_op_oneline(op)}")
                print(f"  {RED}>>>{RESET}  ... {len(inside) - 2 * context} more in this partition ...")
                for op in inside[-context:]:
                    print(f"  {RED}>>>{RESET}  op {op['index']} {_fmt_op_oneline(op)}")

            for j in range(end_pos + 1, min(len(ops), end_pos + 1 + context)):
                print(f"  next: op {ops[j]['index']} {_fmt_op_oneline(ops[j])}")
            print()


# ---------------------------------------------------------------------------
# Comparison harness - predicted vs actual
# ---------------------------------------------------------------------------

def compare_to_actual(results: List[PartitionResult],
                      actual: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Diff simulator output against real apply_plugin_main results.

    `actual` schema (per subgraph signature):
        {"npu_partitions": int, "cpu_partitions": int, "cpu_op_indices": [int, ...]}

    Returns per-subgraph diffs with divergent_ops / false_cpu_ops / agreement.
    """
    diffs: List[Dict[str, Any]] = []
    for r in results:
        predicted_cpu: Set[int] = set()
        predicted_npu: Set[int] = set()
        for p in r.partitions:
            target = predicted_cpu if p.kind == "CPU" else predicted_npu
            for idx in range(p.op_indices[0], p.op_indices[1] + 1):
                target.add(idx)

        a = actual.get(r.subgraph, {})
        actual_cpu = set(a.get("cpu_op_indices", []))
        all_ops = predicted_cpu | predicted_npu
        actual_npu = all_ops - actual_cpu

        divergent = sorted(predicted_npu & actual_cpu)
        false_cpu = sorted(predicted_cpu & actual_npu)
        agree = (len(all_ops) - len(divergent) - len(false_cpu)) / max(1, len(all_ops))

        diffs.append({
            "signature": r.subgraph,
            "predicted": {"npu": sum(1 for p in r.partitions if p.kind == "NPU"),
                          "cpu": sum(1 for p in r.partitions if p.kind == "CPU")},
            "actual": {"npu": a.get("npu_partitions"), "cpu": a.get("cpu_partitions")},
            "divergent_ops": divergent,
            "false_cpu_ops": false_cpu,
            "agreement": agree,
        })
    return diffs


def report_comparison(diffs: List[Dict[str, Any]], show_op_limit: int = 20) -> None:
    for d in diffs:
        print(f"Signature: {d['signature']}")
        print(f"  predicted: NPU={d['predicted']['npu']}  CPU={d['predicted']['cpu']}")
        print(f"  actual:    NPU={d['actual']['npu']}  CPU={d['actual']['cpu']}")
        print(f"  agreement: {d['agreement'] * 100:.1f}%")
        for label, key, color in (("predicted NPU, actual CPU", "divergent_ops", RED),
                                   ("predicted CPU, actual NPU", "false_cpu_ops", GREEN)):
            ops = d[key]
            if not ops:
                continue
            shown = ops[:show_op_limit]
            more = f" (+{len(ops) - show_op_limit} more)" if len(ops) > show_op_limit else ""
            print(f"  {color}{label}{RESET}: {shown}{more}")
        print()


__all__ = [
    "OpSupport",
    "Partition",
    "PartitionResult",
    "load_op_support",
    "simulate_partition",
    "report_partition",
    "report_partitions",
    "report_seams",
    "compare_to_actual",
    "report_comparison",
]

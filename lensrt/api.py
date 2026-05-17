"""Public API: Static and Runtime."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from .graph_parser import (
    LiteRTLMExtractor,
    TFLiteGraphParser,
    _find_cross_signature_divergence,
    _find_dynamic_shape_ops,
    _find_rank_violations,
    _report_cross_signature_divergence,
    _report_dynamic_shape_ops,
    _report_rank_violations,
)
from .partition_simulator import (
    _load_op_support,
    _report_partitions,
    _report_seams,
    _simulate_partition,
)
from .runtime_analyser import RuntimeAnalyser


def _section(title: str, first: bool = False) -> None:
    if not first:
        print()
    print(title)


def _jsonable(obj: Any) -> Any:
    """Recursively convert tuples/Counters/sets to JSON-safe primitives."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, Counter):
        return {str(k): v for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_jsonable(v) for v in obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class Static:
    """Static partition analysis from a .tflite (or .litertlm) model."""

    def __init__(self, model: str, op_support: str):
        self.model_path = model
        self._op_support_path = op_support
        self.graph: Dict[str, Any] = TFLiteGraphParser().parse(model)
        self.op_support = _load_op_support(op_support)
        self.partitions = _simulate_partition(self.graph, self.op_support)

    @classmethod
    def from_litertlm(
        cls, model: str, op_support: str, dump_dir: str = "./dump"
    ) -> "Static":
        extracted = LiteRTLMExtractor.extract(model, dump_dir)
        if not extracted:
            raise RuntimeError(f"No .tflite sections extracted from: {model}")
        return cls(extracted[0], op_support)

    def report(self) -> None:
        _section("Rank violations", first=True)
        _report_rank_violations(self.graph)
        _section("Dynamic shape / index inputs")
        _report_dynamic_shape_ops(self.graph)
        _section("Cross-signature divergence")
        _report_cross_signature_divergence(self.graph)
        _section("Partition simulation")
        _report_partitions(self.partitions)
        _section("Non-delegated seams (+/-2 ops around each non-delegated partition)")
        _report_seams(self.graph, self.partitions, context=2, kind="non_delegated")

    def json(self) -> Dict[str, Any]:
        partitions = []
        for pr in self.partitions:
            partitions.append(
                {
                    "subgraph": pr.subgraph,
                    "total_ops": pr.total_ops,
                    "partitions": [
                        {
                            "kind": p.kind,
                            "op_indices": list(p.op_indices),
                            "op_count": p.op_count,
                            "reason": p.reason,
                            "op_breakdown": dict(p.op_breakdown),
                        }
                        for p in pr.partitions
                    ],
                }
            )
        return _jsonable(
            {
                "model_path": self.model_path,
                "rank_violations": _find_rank_violations(self.graph),
                "dynamic_shapes": _find_dynamic_shape_ops(self.graph),
                "cross_signature_divergence": _find_cross_signature_divergence(
                    self.graph
                ),
                "partitions": partitions,
            }
        )


class Runtime:
    """Runtime partition analysis via apply_plugin_main + rewritten flatbuffer."""

    def __init__(
        self,
        model: str,
        plugin: str,
        tool: str,
        soc: str,
        qnn_lib: str = "",
        out: str = "./runtime_out",
    ):
        if not Path(tool).exists():
            raise RuntimeError(
                f"apply_plugin_main not found: {tool}\n"
                "Build it with: bazel build //litert/tools:apply_plugin_main"
            )
        self.model_path = model
        self._analyser = RuntimeAnalyser(tool, plugin, soc, qnn_lib)
        rewritten, log_path = self._analyser.run(model, out)
        self.results = self._analyser.parse(rewritten, log_path)
        self.rewritten_path = rewritten
        self.log_path = log_path

    def report(self) -> None:
        self._analyser.report(self.results)

    def json(self) -> Dict[str, Any]:
        return _jsonable(
            {
                "model_path": self.model_path,
                "rewritten_path": self.rewritten_path,
                "log_path": self.log_path,
                "subgraphs": [
                    {
                        "subgraph": r.subgraph,
                        "total_ops": r.total_ops,
                        "delegated_op_count": r.delegated_op_count,
                        "non_delegated_op_count": r.non_delegated_op_count,
                        "delegated_op_indices": r.delegated_op_indices,
                        "non_delegated_op_indices": r.non_delegated_op_indices,
                        "non_delegated_op_names": dict(r.non_delegated_op_names),
                    }
                    for r in self.results
                ],
                "global": {
                    "error_codes": dict(self._analyser._global_error_hist),
                    "rejected_op_types": dict(self._analyser._global_type_hist),
                },
            }
        )

"""Parse LiteRT-LM / TFLite graphs and surface partition-relevant signals."""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Any, Dict, List


GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# TFLite parser
# ---------------------------------------------------------------------------

def _enum_map(module) -> Dict[int, str]:
    return {getattr(module, k): k for k in dir(module)
            if k.isupper() and isinstance(getattr(module, k), int)}


def _decode(s) -> str:
    return s.decode() if isinstance(s, (bytes, bytearray)) else str(s)


def _placeholder_tensor(tensor_index: int) -> Dict[str, Any]:
    return {"name": f"<corrupt@{tensor_index}>", "dtype": "UNKNOWN", "shape": [],
            "is_constant": False, "const_values": None, "tensor_index": tensor_index}


class TFLiteGraphParser:
    """Parse a .tflite file into a dict of {path, subgraphs:[{name, ops:[...]}]}."""

    def parse(self, path: str) -> Dict[str, Any]:
        try:
            from tflite.Model import Model
            import tflite.BuiltinOperator as BuiltinOperator
            import tflite.TensorType as TensorType
        except ImportError as e:
            raise RuntimeError("Install `tflite` and `flatbuffers`.") from e

        enum_op = _enum_map(BuiltinOperator)
        enum_type = _enum_map(TensorType)

        with open(path, "rb") as f:
            model = Model.GetRootAsModel(f.read(), 0)

        result: Dict[str, Any] = {"path": path, "subgraphs": []}
        for si in range(model.SubgraphsLength()):
            sg = model.Subgraphs(si)
            name = _decode(sg.Name()) if sg.Name() else "main"
            ops = [self._parse_op(model, sg, oi, enum_op, enum_type)
                   for oi in range(sg.OperatorsLength())]
            result["subgraphs"].append({"name": name, "ops": ops})
        return result

    @staticmethod
    def _parse_op(model, sg, oi, enum_op, enum_type) -> Dict[str, Any]:
        op = sg.Operators(oi)
        opc = model.OperatorCodes(op.OpcodeIndex())
        opname = enum_op.get(opc.BuiltinCode(), f"BUILTIN_{opc.BuiltinCode()}")

        # Tensor parsing is wrapped because the heuristic .litertlm extractor
        # can produce flatbuffers with corrupt tensor offsets near section
        # boundaries. A placeholder entry preserves slot positions for callers.
        inputs = []
        for ti in range(op.InputsLength()):
            idx = op.Inputs(ti)
            try:
                inputs.append(TFLiteGraphParser._parse_input(model, sg, idx, enum_type))
            except Exception:
                inputs.append(_placeholder_tensor(idx))

        outputs = []
        for ti in range(op.OutputsLength()):
            idx = op.Outputs(ti)
            try:
                outputs.append(TFLiteGraphParser._parse_output(sg, idx, enum_type))
            except Exception:
                outputs.append(_placeholder_tensor(idx))

        return {"index": oi, "opname": opname, "inputs": inputs, "outputs": outputs}

    @staticmethod
    def _parse_input(model, sg, ti_idx, enum_type) -> Dict[str, Any]:
        t = sg.Tensors(ti_idx)
        dtype = enum_type.get(t.Type(), str(t.Type()))
        shape = [t.Shape(s) for s in range(t.ShapeLength())]
        buf = model.Buffers(t.Buffer())
        is_const = buf.DataLength() > 0

        const_values = None
        if is_const and dtype == "INT32":
            raw = bytes(buf.DataAsNumpy().tobytes())
            const_values = list(struct.unpack_from(f"<{len(raw) // 4}i", raw))

        return {
            "name": _decode(t.Name()),
            "dtype": dtype,
            "shape": shape,
            "is_constant": is_const,
            "const_values": const_values,
            "tensor_index": ti_idx,
        }

    @staticmethod
    def _parse_output(sg, ti_idx, enum_type) -> Dict[str, Any]:
        t = sg.Tensors(ti_idx)
        return {
            "name": _decode(t.Name()),
            "dtype": enum_type.get(t.Type(), str(t.Type())),
            "shape": [t.Shape(s) for s in range(t.ShapeLength())],
            "tensor_index": ti_idx,
        }


# ---------------------------------------------------------------------------
# .litertlm extractor — heuristic scan for embedded TFL3 blobs
# ---------------------------------------------------------------------------

class LiteRTLMExtractor:
    """Extract .tflite blobs from a .litertlm file by scanning for TFL3 magic."""

    TFLITE_MAGIC = b"TFL3"

    @staticmethod
    def extract(litertlm_path: str, out_dir: str) -> List[str]:
        with open(litertlm_path, "rb") as f:
            content = f.read()
        filesize = len(content)
        os.makedirs(out_dir, exist_ok=True)

        extracted: List[str] = []
        offset = 0
        model_num = 0
        while True:
            pos = content.find(LiteRTLMExtractor.TFLITE_MAGIC, offset)
            if pos == -1:
                break

            # Scan back up to 100 bytes for a plausible flatbuffer root offset.
            start = max(0, pos - 100)
            for back in range(100, 0, -1):
                test = pos - back
                if test < 0:
                    continue
                root_rel = struct.unpack("<I", content[test:test + 4])[0]
                if 0 < root_rel < 1000:
                    start = test
                    break

            next_magic = content.find(LiteRTLMExtractor.TFLITE_MAGIC, pos + 1)
            end = next_magic - 100 if next_magic > 0 else filesize

            if 1000 < end - start < 10_000_000_000:
                model_num += 1
                fp = os.path.join(out_dir, f"Section{model_num}_TFLiteModel_heuristic.tflite")
                with open(fp, "wb") as fout:
                    fout.write(content[start:end])
                extracted.append(fp)
            offset = pos + 1
        return extracted


def extract_and_parse(litertlm_path: str, out_dir: str) -> List[Dict[str, Any]]:
    parser = TFLiteGraphParser()
    return [parser.parse(p) for p in LiteRTLMExtractor.extract(litertlm_path, out_dir)]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _clip(text: str, limit: int = 30) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


def format_tensor_list(tensors: List[Dict[str, Any]], minimize: bool = False) -> str:
    formatted = [f"{RED}{t['name']}{RESET} {RED}{t['dtype']}{RESET} {t['shape']}" for t in tensors]
    if minimize:
        formatted = [_clip(e, 30) for e in formatted]
        if len(formatted) > 2:
            formatted = formatted[:2] + [f"... ({len(formatted) - 2} more)"]
    return ", ".join(formatted)


def pretty_graph(graph: Dict[str, Any], minimize: bool = False, max_ops: int | None = None) -> None:
    print(graph["path"])
    for sg in graph["subgraphs"]:
        print(sg["name"])
        ops = sg["ops"]
        if max_ops and len(ops) > max_ops:
            head = max_ops // 2
            tail = max_ops - head
            print_ops = ops[:head] + [None] + ops[-tail:]
            gap = len(ops) - head - tail
        else:
            print_ops = ops
            gap = 0

        for op in print_ops:
            if op is None:
                print(f"  [... {gap} more here ...]")
                continue
            print(f"{op['index']} {GREEN}{op['opname']}{RESET}")
            print(f"  in:  {format_tensor_list(op['inputs'], minimize)}")
            print(f"  out: {format_tensor_list(op['outputs'], minimize)}")


# ---------------------------------------------------------------------------
# Dynamic-shape / index detection
# ---------------------------------------------------------------------------

# Ops whose listed input slots carry shape/index data. Those slots must be
# constants for the op to be statically resolvable on an NPU.
_SHAPE_INDEX_SLOTS: Dict[str, List[tuple]] = {
    "RESHAPE":                 [(1, "new_shape")],
    "PAD":                     [(1, "paddings")],
    "PADV2":                   [(1, "paddings")],
    "MIRROR_PAD":              [(1, "paddings")],
    "STRIDED_SLICE":           [(1, "begin"), (2, "end"), (3, "strides")],
    "SLICE":                   [(1, "begin"), (2, "size")],
    "GATHER":                  [(1, "indices")],
    "GATHER_ND":               [(1, "indices")],
    "SCATTER_ND":              [(0, "indices")],
    "BROADCAST_TO":            [(1, "shape")],
    "EXPAND_DIMS":             [(1, "axis")],
    "TILE":                    [(1, "multiples")],
    "TRANSPOSE":               [(1, "perm")],
    "RESIZE_BILINEAR":         [(1, "size")],
    "RESIZE_NEAREST_NEIGHBOR": [(1, "size")],
}


def find_dynamic_shape_ops(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return ops whose shape/index inputs are runtime tensors or contain -1.

    Each result entry: {subgraph, op_index, opname, dynamic_inputs: [...]}.
    """
    results = []
    for sg in graph["subgraphs"]:
        for op in sg["ops"]:
            slots = _SHAPE_INDEX_SLOTS.get(op["opname"])
            if slots is None:
                continue

            dynamic = []
            for slot, label in slots:
                if slot >= len(op["inputs"]):
                    continue
                t = op["inputs"][slot]
                if not t["is_constant"]:
                    dynamic.append({"slot": slot, "label": label, "tensor": t, "reason": "runtime"})
                elif op["opname"] == "RESHAPE" and t["const_values"] and -1 in t["const_values"]:
                    dynamic.append({"slot": slot, "label": label, "tensor": t,
                                    "reason": "inferred_dim", "const_values": t["const_values"]})

            if dynamic:
                results.append({
                    "subgraph": sg["name"],
                    "op_index": op["index"],
                    "opname": op["opname"],
                    "dynamic_inputs": dynamic,
                })
    return results


def report_dynamic_shape_ops(graph: Dict[str, Any]) -> None:
    hits = find_dynamic_shape_ops(graph)
    if not hits:
        print(f"{GREEN}No dynamic shape/index ops found in {graph['path']}{RESET}")
        return
    print(f"{RED}{len(hits)} fragmentation candidate(s) in {graph['path']}{RESET}")
    for h in hits:
        print(f"  [{h['op_index']:>4}] {GREEN}{h['opname']}{RESET}  (subgraph: {h['subgraph']})")
        for d in h["dynamic_inputs"]:
            t = d["tensor"]
            if d["reason"] == "inferred_dim":
                print(f"         slot {d['slot']} ({d['label']}): {RED}INFERRED_DIM{RESET}  "
                      f"{t['name']}  {t['dtype']}  values={d['const_values']}")
            else:
                print(f"         slot {d['slot']} ({d['label']}): {RED}RUNTIME{RESET}  "
                      f"{t['name']}  {t['dtype']}  {t['shape']}")


# ---------------------------------------------------------------------------
# Op-type histogram
# ---------------------------------------------------------------------------

def op_histogram(graph: Dict[str, Any]) -> List[tuple]:
    """Return [(opname, count, {dtype: count}), ...] sorted by count desc."""
    totals: Counter = Counter()
    by_dtype: Dict[str, Counter] = defaultdict(Counter)
    for sg in graph["subgraphs"]:
        for op in sg["ops"]:
            dtype = op["outputs"][0]["dtype"] if op["outputs"] else "UNKNOWN"
            totals[op["opname"]] += 1
            by_dtype[op["opname"]][dtype] += 1
    return [(name, count, dict(by_dtype[name])) for name, count in totals.most_common()]


def report_op_histogram(graph: Dict[str, Any], top: int | None = None) -> None:
    hist = op_histogram(graph)
    if not hist:
        print("No ops found.")
        return
    if top:
        hist = hist[:top]

    total = sum(c for _, c, _ in hist)
    max_count = hist[0][1]
    name_w = max(len(name) for name, _, _ in hist)
    count_w = len(str(max_count))
    bar_width = 36

    print(f"Op histogram — {graph['path']}  (total ops: {total})")
    for name, count, dtypes in hist:
        bar = "█" * round(count / max_count * bar_width)
        pct = count / total * 100
        dtype_str = "  ".join(f"{dt}:{n}" for dt, n in sorted(dtypes.items(), key=lambda x: -x[1]))
        print(f"  {GREEN}{name:<{name_w}}{RESET}  {count:>{count_w}}  {bar}  {pct:5.1f}%  {dtype_str}")


__all__ = [
    "TFLiteGraphParser",
    "LiteRTLMExtractor",
    "extract_and_parse",
    "format_tensor_list",
    "pretty_graph",
    "find_dynamic_shape_ops",
    "report_dynamic_shape_ops",
    "op_histogram",
    "report_op_histogram",
]

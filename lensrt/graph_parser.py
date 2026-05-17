"""Parse LiteRT-LM / TFLite graphs and surface partition-relevant signals."""

from __future__ import annotations

import os
import struct
import sys
from typing import Any, Dict, List

_TTY = sys.stdout.isatty()
GREEN = "\033[92m" if _TTY else ""
RED = "\033[91m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""


def _enum_map(module) -> Dict[int, str]:
    return {
        getattr(module, k): k
        for k in dir(module)
        if k.isupper() and isinstance(getattr(module, k), int)
    }


def _decode(s) -> str:
    return s.decode() if isinstance(s, (bytes, bytearray)) else str(s)


def _placeholder_tensor(tensor_index: int) -> Dict[str, Any]:
    return {
        "name": f"<corrupt@{tensor_index}>",
        "dtype": "UNKNOWN",
        "shape": [],
        "is_constant": False,
        "const_values": None,
        "tensor_index": tensor_index,
    }


def _clip(text: str, limit: int = 30) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


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
            name = _decode(sg.Name()) if sg.Name() else f"subgraph_{si}"
            ops = [
                self._parse_op(model, sg, oi, enum_op, enum_type)
                for oi in range(sg.OperatorsLength())
            ]
            result["subgraphs"].append({"name": name, "ops": ops})
        return result

    @staticmethod
    def _parse_op(model, sg, oi, enum_op, enum_type) -> Dict[str, Any]:
        op = sg.Operators(oi)
        opc = model.OperatorCodes(op.OpcodeIndex())
        opname = enum_op.get(opc.BuiltinCode(), f"BUILTIN_{opc.BuiltinCode()}")

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

            start = max(0, pos - 100)
            for back in range(100, 0, -1):
                test = pos - back
                if test < 0:
                    continue
                root_rel = struct.unpack("<I", content[test : test + 4])[0]
                if 0 < root_rel < 1000:
                    start = test
                    break

            next_magic = content.find(LiteRTLMExtractor.TFLITE_MAGIC, pos + 1)
            end = next_magic - 100 if next_magic > 0 else filesize

            if 1000 < end - start < 10_000_000_000:
                model_num += 1
                fp = os.path.join(
                    out_dir, f"Section{model_num}_TFLiteModel_heuristic.tflite"
                )
                with open(fp, "wb") as fout:
                    fout.write(content[start:end])
                extracted.append(fp)
            offset = pos + 1
        return extracted


_SHAPE_INDEX_SLOTS: Dict[str, List[tuple]] = {
    "RESHAPE": [(1, "new_shape")],
    "PAD": [(1, "paddings")],
    "PADV2": [(1, "paddings")],
    "MIRROR_PAD": [(1, "paddings")],
    "STRIDED_SLICE": [(1, "begin"), (2, "end"), (3, "strides")],
    "SLICE": [(1, "begin"), (2, "size")],
    "GATHER": [(1, "indices")],
    "GATHER_ND": [(1, "indices")],
    "SCATTER_ND": [(0, "indices")],
    "BROADCAST_TO": [(1, "shape")],
    "EXPAND_DIMS": [(1, "axis")],
    "TILE": [(1, "multiples")],
    "TRANSPOSE": [(1, "perm")],
    "RESIZE_BILINEAR": [(1, "size")],
    "RESIZE_NEAREST_NEIGHBOR": [(1, "size")],
}

_OP_RANK_LIMITS: Dict[str, int] = {
    "CONV_2D": 4,
    "DEPTHWISE_CONV_2D": 4,
    "CONV_3D": 5,
    "BATCH_MATMUL": 5,
    "FULLY_CONNECTED": 4,
    "SOFTMAX": 4,
    "REDUCE_MAX": 5,
    "REDUCE_MIN": 5,
    "REDUCE_PROD": 5,
    "REDUCE_SUM": 5,
}


def _find_dynamic_shape_ops(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    inferred_dim_ops = {"RESHAPE", "PAD", "PADV2", "BROADCAST_TO", "TILE"}
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
                    dynamic.append(
                        {"slot": slot, "label": label, "tensor": t, "reason": "runtime"}
                    )
                elif (
                    op["opname"] in inferred_dim_ops
                    and t["const_values"]
                    and -1 in t["const_values"]
                ):
                    dynamic.append(
                        {
                            "slot": slot,
                            "label": label,
                            "tensor": t,
                            "reason": "inferred_dim",
                            "const_values": t["const_values"],
                        }
                    )

            if dynamic:
                results.append(
                    {
                        "subgraph": sg["name"],
                        "op_index": op["index"],
                        "opname": op["opname"],
                        "dynamic_inputs": dynamic,
                    }
                )
    return results


def _report_dynamic_shape_ops(
    graph: Dict[str, Any], max_name: int = 40, max_hits: int | None = None
) -> None:
    hits = _find_dynamic_shape_ops(graph)
    if not hits:
        print(f"{GREEN}No dynamic shape/index ops found in {graph['path']}{RESET}")
        return
    print(f"{RED}{len(hits)} fragmentation candidate(s) in {graph['path']}{RESET}")
    shown = hits[:max_hits] if max_hits else hits
    for h in shown:
        sg = _clip(h["subgraph"], max_name)
        print(f"  [{h['op_index']:>4}] {GREEN}{h['opname']}{RESET}  (subgraph: {sg})")
        for d in h["dynamic_inputs"]:
            t = d["tensor"]
            name = _clip(t["name"], max_name)
            if d["reason"] == "inferred_dim":
                print(
                    f"         slot {d['slot']} ({d['label']}): {RED}INFERRED_DIM{RESET}  "
                    f"{name}  {t['dtype']}  values={d['const_values']}"
                )
            else:
                print(
                    f"         slot {d['slot']} ({d['label']}): {RED}RUNTIME{RESET}  "
                    f"{name}  {t['dtype']}  {t['shape']}"
                )
    if max_hits and len(hits) > max_hits:
        print(f"  ... ({len(hits) - max_hits} more)")


def _find_rank_violations(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    for sg in graph["subgraphs"]:
        for op in sg["ops"]:
            limit = _OP_RANK_LIMITS.get(op["opname"])
            if limit is None:
                continue

            violations = []
            for input_idx, t in enumerate(op["inputs"]):
                rank = len(t["shape"])
                if rank > limit:
                    violations.append(
                        {
                            "input_index": input_idx,
                            "tensor": t,
                            "rank": rank,
                            "limit": limit,
                        }
                    )

            if violations:
                results.append(
                    {
                        "subgraph": sg["name"],
                        "op_index": op["index"],
                        "opname": op["opname"],
                        "violations": violations,
                    }
                )
    return results


def _report_rank_violations(
    graph: Dict[str, Any], max_name: int = 40, max_hits: int | None = None
) -> None:
    hits = _find_rank_violations(graph)
    if not hits:
        print(f"{GREEN}No rank violations found in {graph['path']}{RESET}")
        return
    print(f"{RED}{len(hits)} rank violation(s) in {graph['path']}{RESET}")
    shown = hits[:max_hits] if max_hits else hits
    for h in shown:
        sg = _clip(h["subgraph"], max_name)
        print(f"  [{h['op_index']:>4}] {GREEN}{h['opname']}{RESET}  (subgraph: {sg})")
        for v in h["violations"]:
            t = v["tensor"]
            name = _clip(t["name"], max_name)
            print(
                f"         input {v['input_index']}: {RED}RANK {v['rank']} > {v['limit']}{RESET}  "
                f"{name}  {t['dtype']}  shape={t['shape']}"
            )
    if max_hits and len(hits) > max_hits:
        print(f"  ... ({len(hits) - max_hits} more)")


def _find_cross_signature_divergence(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    if len(graph["subgraphs"]) < 2:
        return []

    dynamic_ops = _find_dynamic_shape_ops(graph)
    divergence_map: Dict[int, Dict[str, Any]] = {}

    for entry in dynamic_ops:
        op_idx = entry["op_index"]
        opname = entry["opname"]
        subgraph = entry["subgraph"]

        if op_idx not in divergence_map:
            divergence_map[op_idx] = {
                "opname": opname,
                "subgraphs": set(),
            }
        divergence_map[op_idx]["subgraphs"].add(subgraph)

    results = []
    for op_idx, data in divergence_map.items():
        if len(data["subgraphs"]) > 1:
            results.append(
                {
                    "op_index": op_idx,
                    "opname": data["opname"],
                    "subgraphs": sorted(list(data["subgraphs"])),
                }
            )

    return results


def _report_cross_signature_divergence(
    graph: Dict[str, Any], max_name: int = 40, max_hits: int | None = None
) -> None:
    hits = _find_cross_signature_divergence(graph)
    if not hits:
        print(f"{GREEN}No cross-signature divergence found in {graph['path']}{RESET}")
        return
    print(
        f"{RED}{len(hits)} cross-signature divergence candidate(s) in {graph['path']}{RESET}"
    )
    shown = hits[:max_hits] if max_hits else hits
    for h in shown:
        print(
            f"  [{h['op_index']:>4}] {GREEN}{h['opname']}{RESET}  appears in {len(h['subgraphs'])} subgraph(s)"
        )
        for sg in h["subgraphs"]:
            sg_clip = _clip(sg, max_name)
            print(f"         {sg_clip}")
    if max_hits and len(hits) > max_hits:
        print(f"  ... ({len(hits) - max_hits} more)")

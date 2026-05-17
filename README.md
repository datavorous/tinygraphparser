# tinygraphparser

Parse TFLite / LiteRT-LM graphs and statically predict how the LiteRT QNN
delegate will partition them between NPU and CPU.

Built for Gemma-family edge inference analysis. Works on any `.tflite` or
`.litertlm` file.

## Install

```
uv sync
```

## Quick start

```python
from tinygraphparser import (
    LiteRTLMExtractor, TFLiteGraphParser,
    report_op_histogram, report_dynamic_shape_ops,
    load_op_support, simulate_partition, report_partitions, report_seams,
)

tflite_files = LiteRTLMExtractor.extract("model.litertlm", "./dump")
graph        = TFLiteGraphParser().parse(tflite_files[0])
op_support   = load_op_support("analysis/opSupportMap.csv")
partitions   = simulate_partition(graph, op_support)

report_op_histogram(graph)
report_dynamic_shape_ops(graph)
report_partitions(partitions)
report_seams(graph, partitions, context=2)
```

---

## API

### `LiteRTLMExtractor`

```python
LiteRTLMExtractor.extract(litertlm_path: str, out_dir: str) -> list[str]
```

Heuristically scans a `.litertlm` for embedded `TFL3` magic bytes and writes
each found blob to `out_dir` as `Section{N}_TFLiteModel_heuristic.tflite`.
Returns the list of written paths.

---

### `TFLiteGraphParser`

```python
graph = TFLiteGraphParser().parse(path: str) -> dict
```

Parses a `.tflite` flatbuffer. Returns:

```python
{
  "path": str,
  "subgraphs": [{
    "name": str,
    "ops": [{
      "index":   int,
      "opname":  str,           # "FULLY_CONNECTED", "RESHAPE", etc.
      "inputs":  [tensor, ...],
      "outputs": [tensor, ...],
    }]
  }]
}
```

Each **input** tensor:

```python
{
  "name":         str,
  "dtype":        str,               # "FLOAT32", "INT8", "INT32", "BOOL", ...
  "shape":        list[int],
  "is_constant":  bool,              # True if backed by a non-empty flatbuffer buffer
  "const_values": list[int] | None,  # decoded only for INT32 constants
  "tensor_index": int,
}
```

**Output** tensors carry `name`, `dtype`, `shape`, `tensor_index` — no
constness, outputs are never compile-time constants.

---

### `extract_and_parse`

```python
extract_and_parse(litertlm_path: str, out_dir: str) -> list[dict]
```

Convenience: extract all blobs then parse each one. Returns graph dicts in
section order.

---

### `pretty_graph`

```python
pretty_graph(graph: dict, minimize: bool = False, max_ops: int | None = None)
```

Prints a human-readable op listing. `max_ops` shows the first and last N/2 ops
with a gap indicator. `minimize=True` clips long tensor names.

---

### `op_histogram` / `report_op_histogram`

```python
op_histogram(graph: dict, top: int | None = None) -> list[tuple]
# returns [(opname, total_count, {dtype: count}), ...]

report_op_histogram(graph: dict, top: int | None = None)
```

Counts ops by type, sorted descending. Each row is split by output dtype —
useful for spotting mixed-precision hotspots (e.g. `MUL  FLOAT32:351  INT32:36`).

---

### `find_dynamic_shape_ops` / `report_dynamic_shape_ops`

```python
find_dynamic_shape_ops(graph: dict) -> list[dict]
report_dynamic_shape_ops(graph: dict)
```

Flags ops whose shape/index inputs are not compile-time constants. These are
fragmentation candidates: the NPU delegate cannot resolve memory layout
statically and must fall back to CPU or defer allocation.

Ops and slots checked:

| Op | Slot |
|----|------|
| RESHAPE | new_shape (1) |
| PAD / PADV2 / MIRROR_PAD | paddings (1) |
| STRIDED_SLICE | begin (1), end (2), strides (3) |
| SLICE | begin (1), size (2) |
| GATHER / GATHER_ND | indices (1) |
| SCATTER_ND | indices (0) |
| BROADCAST_TO / TILE / TRANSPOSE | shape / multiples / perm (1) |
| RESIZE_BILINEAR / RESIZE_NEAREST_NEIGHBOR | size (1) |

Two failure modes are reported separately:

- `runtime` — shape tensor has no buffer data (fully runtime)
- `inferred_dim` — shape is a constant INT32 buffer but contains `-1`
  (RESHAPE only; QNN handles this in most cases but it is still worth tracking)

---

### `load_op_support`

```python
load_op_support(csv_path: str) -> OpSupport
```

Parses the tab-separated `opSupportMap.csv`
(`litert_op_code | qnn_legalization | builder_file | source_line`).

```python
@dataclass
class OpSupport:
    tfl_supported:      set[str]        # TFLite enum names with a QNN builder
    builder_file:       dict[str, str]  # source file per op name
    composite_supported: set[str]       # SHLO composite variants (e.g. "kRmsNorm")
    rows:               list[dict]      # raw parsed rows
```

---

### `simulate_partition` / `report_partitions`

```python
simulate_partition(graph: dict, op_support: OpSupport) -> list[PartitionResult]
report_partitions(results: list[PartitionResult])
```

Classifies each op as NPU-eligible (has a QNN builder **and** no dynamic
shape/index inputs) and groups contiguous runs into partitions.

```python
@dataclass
class Partition:
    kind:         str             # "NPU" | "CPU"
    op_indices:   tuple[int,int]  # inclusive (start, end)
    op_count:     int
    subgraph:     str
    reason:       str | None      # CPU only: "no_builder" | "dynamic_shape" | "unsupported_composite"
    op_breakdown: Counter         # CPU only: opname -> count
```

Sample output:

```
Signature: main (2182 ops)
  Partitions: 1 total - 1 NPU, 0 CPU
  Largest  NPU partition: 2182 ops (ops 0-2181)
  Smallest NPU partition: 2182 ops (ops 0-2181)
  Mean     NPU partition: 2182.0 ops
```

The largest NPU partition size is the headline health metric. One large
partition = one delegate dispatch. Many small ones = frequent NPU/CPU switches.

---

### `report_seams`

```python
report_seams(graph, results, context=2, kind="CPU")
```

For each CPU partition, prints `context` ops before and after it so you can
see exactly what caused the split without re-running.

```
Partition 1 (CPU, 1 op, reason=dynamic_shape)
  prev: op 3 ADD FLOAT32 [1,128,42] -> [1,128,42]
  >>>  op 4 RESHAPE FLOAT32 [1,128,42] -> [1,128,42,1]
  next: op 5 RSQRT FLOAT32 [1,128,42,1] -> [1,128,42,1]
```

---

### `compare_to_actual` / `report_comparison`

```python
compare_to_actual(results, actual: dict[str, dict]) -> list[dict]
report_comparison(diffs: list[dict], show_op_limit=20)
```

Diffs simulator predictions against real `apply_plugin_main` output once you
have it. `actual` schema per subgraph signature:

```python
{
  "npu_partitions": int,
  "cpu_partitions": int,
  "cpu_op_indices": [int, ...],
}
```

Each diff record carries `divergent_ops` (predicted NPU, actually CPU) and
`false_cpu_ops` (predicted CPU, actually NPU), plus an `agreement` fraction.
`divergent_ops` is the interesting class — likely causes are dtype/shape
constraints not in `opSupportMap.csv`, SDK-level rejections, or partition-size
heuristics in the real delegate.

---

## Notes

The `.litertlm` extractor is heuristic: it finds `TFL3` magic and scans back
for a plausible flatbuffer root offset. Section boundaries can be slightly off,
producing a small number of `<corrupt@N>` placeholder tensors at section tails.
The rest of the graph remains usable.

`opSupportMap.csv` was extracted from the LiteRT source tree and reflects
which ops have a registered QNN builder. It does not capture per-op dtype or
shape constraints that the real delegate checks at runtime. The comparison
harness exists to surface that gap.

# tinygraphparser

Parse TFLite / LiteRT-LM graphs and statically predict QNN delegate partitioning (NPU vs CPU).

## Setup

```
uv sync
uv run python examples/main.py
```

## Usage

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

report_op_histogram(graph)          # op counts split by dtype
report_dynamic_shape_ops(graph)     # ops that will fragment partitions
report_partitions(partitions)       # NPU/CPU partition summary
report_seams(graph, partitions)     # context ops around each CPU split
```

## API

| Function | Returns | Notes |
|---|---|---|
| `LiteRTLMExtractor.extract(path, out_dir)` | `list[str]` | heuristic TFL3 scan |
| `TFLiteGraphParser().parse(path)` | `dict` | full graph with tensor constness |
| `extract_and_parse(path, out_dir)` | `list[dict]` | extract + parse in one call |
| `pretty_graph(graph, minimize, max_ops)` | — | print op listing |
| `op_histogram(graph)` | `[(name, count, {dtype: n})]` | sorted by count desc |
| `find_dynamic_shape_ops(graph)` | `list[dict]` | fragmentation candidates |
| `load_op_support(csv_path)` | `OpSupport` | parses opSupportMap.csv |
| `simulate_partition(graph, op_support)` | `list[PartitionResult]` | static NPU/CPU split |
| `report_seams(graph, results, context, kind)` | — | ops around partition boundaries |
| `compare_to_actual(results, actual)` | `list[dict]` | diff against real partitioner output |

### Graph dict shape

```python
graph = {
  "path": str,
  "subgraphs": [{
    "name": str,
    "ops": [{"index": int, "opname": str, "inputs": [...], "outputs": [...]}]
  }]
}
```

Input tensors carry `name`, `dtype`, `shape`, `is_constant`, `const_values` (INT32 only), `tensor_index`.

### Partition health metric

Largest NPU partition size is the key number. One large partition = one delegate dispatch. Many small ones = frequent NPU/CPU switches.

CPU fallback reasons: `no_builder` · `dynamic_shape` · `unsupported_composite`

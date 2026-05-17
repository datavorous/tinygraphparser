# lensrt
TFLite graph analyzer for QNN delegation diagnostics.

## Install

```bash
uv sync
```

## Usage

### Static

```python
from lensrt import Static

s = Static("models/qwen.tflite", "analysis/opSupportMap.csv")
s.report()
data = s.json()
```

For `.litertlm` files:

```python
s = Static.from_litertlm("model.litertlm", "analysis/opSupportMap.csv")
```

Expected output (first signature only):

```
Rank violations
No rank violations found in models/qwen.tflite

Dynamic shape / index inputs
4 fragmentation candidate(s) in models/qwen.tflite
  [  17] GATHER_ND  (subgraph: decode)
         slot 1 (indices): RUNTIME  ai_edge_torch.generative.utilities.co...  INT64  [1, 1]
  [  54] RESHAPE  (subgraph: decode)
         slot 1 (new_shape): RUNTIME  arith.constant204  INT32  [0]

Cross-signature divergence
1 cross-signature divergence candidate(s) in models/qwen.tflite
  [  17] GATHER_ND  appears in 2 subgraph(s)
         decode
         prefill_128

Partition simulation
Signature: decode (1326 ops)
  Partitions: 5 total, 3 delegated, 2 non_delegated
  Largest delegated partition: 1271 ops (ops 55-1325)
  Smallest delegated partition: 17 ops (ops 0-16)
  Mean delegated partition: 441.3 ops
  Non-delegated breakdown:
    dynamic_shape: 2 ops [GATHER_ND x1, RESHAPE x1]
```

### Runtime

```python
from lensrt import Runtime

r = Runtime(
    model="models/qwen.tflite",
    plugin="path/to/libLiteRtCompilerPlugin_Qualcomm.so",
    tool="path/to/apply_plugin_main",
    soc="SM8650",
    qnn_lib="path/to/qairt/lib/x86_64-linux-clang",
    out="./runtime_out",
)
r.report()
data = r.json()
```

Expected output:

```
Signature: decode
  Non-delegated original ops: 125
  Top non-delegated op types:
      121  FULLY_CONNECTED
        1  EMBEDDING_LOOKUP
        1  GREATER_EQUAL
        1  LESS_EQUAL
        1  GATHER_ND

Signature: prefill_128
  Non-delegated original ops: 168
  Top non-delegated op types:
      116  FULLY_CONNECTED
       48  DYNAMIC_UPDATE_SLICE
        1  EMBEDDING_LOOKUP
        1  GREATER_EQUAL
        1  LESS_EQUAL

Global (log-derived)  -- not attributed to any subgraph
  ValidateOp rejection codes:
    3110  dtype_mismatch        586
  Top rejected op types:
      474  FullyConnected
       96  ElementWiseSelect
        8  ElementWiseBinary
        4  Gather
        4  GatherNd
```

## What it checks

### Static (no SDK required)
- Missing QNN builder
- Input rank exceeds QNN cap
- Dynamic shape or index input
- Inferred -1 dim (RESHAPE, PAD, BROADCAST_TO, TILE)
- Cross-signature divergence
- Dtype risk (FLOAT32 on FC/Gather — delegated only if quantized)

### Runtime (requires LiteRT build + QNN SDK)
- Per-subgraph delegated/non_delegated op counts from rewritten flatbuffer
- Global ValidateOp rejection codes from log (not per-subgraph)

## Limitations
- Static checks are necessary but not sufficient for delegation
- Runtime error codes are log-global, not per-subgraph
- Delegated = DISPATCH_OP present; actual backend (HTP/HVX/GPU) determined by QNN runtime, not this tool
- opSupportMap.csv pinned to LiteRT commit b2df679f

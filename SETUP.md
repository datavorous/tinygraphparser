# LensRT - Local Setup Context

This file is for Claude Code (or any engineer picking this up fresh).
It documents the exact environment, paths, versions, and commands
needed to get both static and runtime analysis working from scratch.

---

## Repo layout (relative to `edge/`)

```
edge/
  static_analyser/graphlens/   ← this library
  work/
    LiteRT/                    ← LiteRT source, built with Bazel
  qairt/
    2.46.0.260424/             ← QNN SDK (Qualcomm AI Runtime)
  models/                      ← model files go here (not committed)
```

---

## 1. Python environment

```bash
cd static_analyser/graphlens
uv sync
```

Requires Python 3.10+. Installs `tflite`, `flatbuffers`, and dev deps.
Static analysis works after this step - no SDK needed.

---

## 2. Bazel

LiteRT uses Bazel. The required version is pinned in `work/LiteRT/.bazelversion`:

```
7.7.0
```

Install bazelisk (auto-downloads the right version):

```bash
pip install bazelisk
# or on Arch: yay -S bazelisk
```

Bazel binary should be at `~/.local/bin/bazel` or on PATH.
Verify: `bazel --version` → should print `bazel 7.7.0`.

---

## 3. QNN SDK

Download **Qualcomm AI Runtime Community** (QAIRT) v2.46.0.260424:
- Source: https://softwarecenter.qualcomm.com (requires Qualcomm account)
- File: `v2.46.0.260224.zip` or equivalent

Unzip to:
```
edge/qairt/2.46.0.260424/
```

Expected contents: `bin/`, `include/`, `lib/`, `lib-safe/`, `share/`, `sdk.yaml`

---

## 4. Bazel BUILD file for QAIRT

Bazel needs a `BUILD` and `WORKSPACE` file at the QAIRT root.
Copy the template from LiteRT source:

```bash
cp work/LiteRT/third_party/qairt/qairt.BUILD qairt/2.46.0.260424/BUILD
echo 'workspace(name = "qairt")' > qairt/2.46.0.260424/WORKSPACE
```

The `qairt.BUILD` template declares `qnn_lib_headers` and `exports_files(glob(["**/*.so"]))`.

---

## 5. Build LiteRT binaries

From `edge/work/LiteRT/`:

```bash
QAIRT=$HOME/research/edge/qairt/2.46.0.260424

bazel build //litert/tools:apply_plugin_main \
  --override_repository=qairt=$QAIRT

bazel build //litert/vendors/qualcomm/compiler:qnn_compiler_plugin \
  --override_repository=qairt=$QAIRT
```

Takes ~15–30 min first time (downloads hermetic toolchain + builds deps).
Cached on subsequent runs.

Output binaries:
```
work/LiteRT/bazel-bin/litert/tools/apply_plugin_main                          (~1 MB)
work/LiteRT/bazel-bin/litert/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so
```

---

## 6. Runtime library dependencies

The QNN plugin `.so` is built with Clang and links against Clang's C++ runtime.

Check what's missing:
```bash
ldd work/LiteRT/bazel-bin/litert/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so | grep "not found"
```

Install missing libs:

```bash
# Arch Linux
sudo pacman -Sy libc++

# Ubuntu / Debian
sudo apt install libc++-dev libc++abi-dev libunwind-dev
```

If `libunwind.so.1` is missing but `libunwind.so.8` exists (common on Arch),
symlink it:
```bash
ln -sf /usr/lib/libunwind.so.8 /usr/lib/libunwind.so.1
```

---

## 7. LD_LIBRARY_PATH

Must include the x86_64 Clang QNN libs at runtime:

```bash
export LD_LIBRARY_PATH=$HOME/research/edge/qairt/2.46.0.260424/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
```

Or pass `qnn_lib_dir=` to `Inspector.from_runtime()` - it sets this automatically.

---

## 8. Using the library

```python
from lensrt import Inspector

# Static only
inspector = Inspector.from_tflite(
    "path/to/model.tflite",
    op_support_path="analysis/opSupportMap.csv",
)
inspector.analyze()
inspector.report_partitions()

# Static + Runtime
inspector = Inspector.from_runtime(
    model_path="path/to/model.tflite",
    op_support_path="analysis/opSupportMap.csv",
    apply_plugin_main_path="work/LiteRT/bazel-bin/litert/tools/apply_plugin_main",
    plugin_path="work/LiteRT/bazel-bin/litert/vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so",
    soc="SM8650",
    qnn_lib_dir="qairt/2.46.0.260424/lib/x86_64-linux-clang",
    output_dir="./runtime_out",
)
inspector.report_runtime()
inspector.report_divergence()
```

Output written to `output_dir/`:
- `rewritten.tflite` - flatbuffer with `DISPATCH_OP` nodes (ground truth for placement)
- `run.log` - full apply_plugin_main log including `ValidateOp` rejection reasons

---

## 9. Common SoC strings

| Device             | `soc=`   |
|--------------------|----------|
| Snapdragon 8 Gen 3 | `SM8650` |
| Snapdragon 8 Gen 2 | `SM8550` |
| Snapdragon 888     | `SM8350` |

---

## 10. opSupportMap.csv - keep it current

The CSV is pinned to LiteRT commit `b2df679f`. If LiteRT is updated, re-extract:

```bash
cd work/LiteRT
git log --oneline -1    # check current commit

# Re-extract op builders from source
python3 scripts/extract_op_support.py \
  litert/vendors/qualcomm/compiler/qnn_compose_graph.cc \
  > ../../static_analyser/graphlens/analysis/opSupportMap.csv
```

If the CSV is stale, partition simulation will silently over-predict
delegated eligibility for any ops added or removed in the updated LiteRT.

---

## 11. Verified environment (as of 2026-05-17)

| Component             | Version / Commit          |
|-----------------------|---------------------------|
| OS                    | Arch Linux, kernel 6.19.6 |
| Python                | 3.12                      |
| Bazel                 | 7.7.0                     |
| LiteRT                | commit b2df679f            |
| QAIRT (QNN SDK)       | 2.46.0.260424             |
| Target SoC (tested)   | SM8650 (Snapdragon 8 Gen 3)|

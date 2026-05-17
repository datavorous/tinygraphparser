# Run from the project root: uv run python examples/main.py
# Expects gemma.litertlm two levels up (edge/gemma.litertlm).
from tinygraphparser import (
    LiteRTLMExtractor,
    TFLiteGraphParser,
    pretty_graph,
    report_op_histogram,
    report_dynamic_shape_ops,
    load_op_support,
    simulate_partition,
    report_partitions,
    report_seams,
)


def section(title: str) -> None:
    print(f"\n## {title}\n")


tflite_files = LiteRTLMExtractor.extract("../../gemma.litertlm", "../../litertlm_dump")
graph = TFLiteGraphParser().parse(tflite_files[2])
op_support = load_op_support("../analysis/opSupportMap.csv")
partitions = simulate_partition(graph, op_support)

section("Graph structure (first/last 4 ops)")
pretty_graph(graph, minimize=True, max_ops=8)

section("Op histogram (with dtype splits)")
report_op_histogram(graph)

section("Dynamic shape / index inputs (fragmentation candidates)")
report_dynamic_shape_ops(graph)

section("Partition simulation")
report_partitions(partitions)

section("CPU seams (+/-2 ops around each CPU partition)")
report_seams(graph, partitions, context=2, kind="CPU")

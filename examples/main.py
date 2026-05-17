from lensrt import Static

s = Static.from_litertlm(
    "../../gemma.litertlm",
    "../analysis/opSupportMap.csv",
    dump_dir="../../litertlm_dump",
)
s.report()

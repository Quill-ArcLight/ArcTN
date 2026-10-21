# ArcTN Heavy slicing comparison harness

This executable applies exactly one slicing mechanism to a caller-supplied
Heavy SSA path. It never runs ArcTN path search and never substitutes a Light
path.

The package build layout is:

```text
qc32-final-heavy-slicing-comparison/
├── arctn-source/             # clone of source/arctn-main.bundle, branch main
└── arctn-slicing-harness/
```

Build it with:

```bash
cargo build --release --manifest-path arctn-slicing-harness/Cargo.toml
```

Run one mode per worker process:

```bash
arctn-slicing-harness \
  --mode fixed \
  --network corpus/example.json \
  --heavy-path heavy-result.json \
  --target-size 67108864 \
  --output fixed.json

arctn-slicing-harness \
  --mode dynamic \
  --network corpus/example.json \
  --heavy-path heavy-result.json \
  --target-size 67108864 \
  --output dynamic.json
```

`fixed` calls `arctn::find_slices_to_size`. `dynamic` calls
`arctn::slice::slice_and_reconf_to_size_with_objective` with a pure-FLOPs
objective, `rounds=3`, and `subtree_size=8`. Both outputs contain the full
input Heavy SSA path, final SSA path, sliced legs, native sliced metrics,
target validation, and slicing-stage wall/CPU time. The fixed-path routine has
no objective argument; its output is still evaluated under the frozen
pure-FLOPs comparison objective, while the dynamic routine receives that
objective directly.

The Heavy-path JSON may be either a raw SSA array or an object containing an
`ssa_path` field. If it carries a `method` or `planner.preset` field, the
harness rejects metadata that identifies a non-Heavy ArcTN path.


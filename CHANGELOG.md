# Changelog

## Unreleased

- Add opt-in output-index slicing through `allow_output_slicing=True`; the
  default remains internal-index slicing only. Single-process execution sums
  internal slices within output blocks and assembles the complete output in
  its declared axis order. Saved plans support the same execution behavior.
  The per-slice size target does not limit complete-output storage; MPI remains
  internal-index only.
- Keep compatibility with ABI 1 Light/Heavy engines: when output-index slicing
  is enabled, the public adapter requests an unsliced order and applies the
  public slicing implementation locally, without changing the ABI 1 request format.
- Add independent, opt-in `integer-order-dp` and `integer-tree-cost` Cargo features
  for fixed-leaf-order DP and contraction-tree cost calculations, including local
  subtree replacement comparisons and cost-cache updates. Both are disabled by
  default and use checked `u128` costs for dimension-two networks with supported
  objectives. Unsupported or unrepresentable cases retain the previous arithmetic.
  The fixed-order DP can skip overflowing candidates if a complete solution fits;
  tree search falls back and evaluates the same candidate with the previous arithmetic.
- These experimental features preserve the mathematical acceptance-probability
  definitions, but exact comparisons and rounding can change decisions, random-number
  consumption, and search trajectories. Existing comparisons include both better
  and worse final objective values; enabling them does not guarantee better paths.
  Python source builds expose matching Cargo features; a separately built Light/Heavy
  engine must select its own features independently.
- Use exact `u128` costs in subset dynamic programming, including local subtree
  reconfiguration, when every index has dimension two and a conservative bound
  proves that all candidate costs fit. Supported objectives are total FLOPs,
  total logical read/write, and the default `FLOPs + 64 * read/write` objective.
  Other dimensions, unsupported mixed weights, and costs outside the bound retain
  the previous arithmetic. Public metrics keep their logarithmic format.
- Integer comparisons can select a different contraction path when floating-point
  scores previously tied or lost low-order cost differences. This change does not
  alter the allowed subset partitions, fixed-leaf-order DP, or tree-search policy.

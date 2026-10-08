# Changelog

## Unreleased

- Use exact `u128` costs in subset dynamic programming, including local subtree
  reconfiguration, when every index has dimension two and a conservative bound
  proves that all candidate costs fit. Supported objectives are total FLOPs,
  total logical read/write, and the default `FLOPs + 64 * read/write` objective.
  Other dimensions, unsupported mixed weights, and costs outside the bound retain
  the previous arithmetic. Public metrics keep their logarithmic format.
- Integer comparisons can select a different contraction path when floating-point
  scores previously tied or lost low-order cost differences. This change does not
  alter the allowed subset partitions, fixed-leaf-order DP, or tree-search policy.

# Prospective edge-case clarification

This clarification is committed before source freeze and before any new contribution or deletion
score. The scientific reviewer approved the protocol in substance and requested these explicit
definitions to make independent reconstruction unambiguous.

- An active token has positive original pooling occupancy in at least one QRS region. Zero signed
  attribution does not remove an active token from the concentration denominator.
- The top-positive-quarter set contains only strictly positive contributions among the first
  `ceil(B/4)` entries in signed-descending, original-index tie order. It may have fewer entries.
  Jaccard is null when both sets are empty; otherwise its union is the denominator.
- Random-deletion seed strings format each fraction with exactly two decimal digits:
  `0.10`, `0.25` and `0.50`.
- Population summaries average defined values within each patient across the three original banks
  first. Report how many banks and patients have undefined values. Bank consistency first averages
  eligible defined comparisons within each patient; patients without an eligible defined comparison
  remain counted as undefined. An empty clinical review packet is never interpreted as a label.

These definitions do not change the primary engineering gate or add a clinical endpoint.

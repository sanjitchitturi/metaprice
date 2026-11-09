# Paper reproduction

This repository implements the method in *MetaPrice: Learning to Price with
Few Samples via Hierarchical Meta-Learning*.

## Method mapping

- Section 3.3 maps each task to category, price-tier, and elasticity learners.
- Equations 3 and 4 are implemented as attention-weighted parameter composition.
- Section 3.5 trains hierarchy learners first, freezes them, then trains the
  attention network on validation tasks through composed initialization and
  inner-loop adaptation.
- Section 3.6 adapts one composed model for a new product and selects the price
  with maximum predicted revenue.

The original paper is authoritative. Section 4.2 is used when the paper's
dataset eligibility statements conflict: products with at least three observed
price-demand pairs are interpolated to 50 points. Existing configuration values
resolve implementation details the paper does not specify.

The paper does not specify a normalization transform, the exact product split,
or the human-reviewed category labels. This implementation applies per-product
min-max normalization and keyword-only category assignment. Its seeded product
split is reproducible from the input workbook, but it cannot be asserted to be
the original split without the authors' split manifest. These choices can change
both the hierarchy membership and the reported metrics.

| Paper statement | Implemented interpretation | Evidence gap |
| --- | --- | --- |
| Section 4.1 requires at least 50 observed price-demand pairs; Section 4.2 interpolates products with fewer than 50 | Admit at least three observed pairs and interpolate to 50 | Original eligibility rule is internally inconsistent |
| Categories use keyword matching and human validation | Deterministic keyword matching only | Human-reviewed labels are unavailable |
| Test-time hierarchy assignment avoids query data | Elasticity and price tier use the support slice, but the slice and its per-product scaling are derived from the complete interpolated curve | Query-side information can still influence the task representation |
| Fixed seed and publicly available split | Seed 42 and product-disjoint random split | Original product assignments are unavailable |

## Evaluation boundary

The product split keeps test products out of model training. Within each product,
the code aggregates and interpolates all observed prices and demands, then
normalizes the complete curve before dividing its sorted points into support and
query sets. Query prices and demands influence interpolation and normalization;
the support prices used for price-tier assignment are themselves interpolated
over the complete observed price range. The reported query metrics describe
this curve reconstruction protocol; they are not estimates
of performance on unseen future transactions. Do not describe this split as a
leakage-free transaction holdout or interpret the revenue ratio as realized uplift.

A separate prospective evaluation would need to withhold raw transactions before
fitting any per-product transformation and define a feasible price intervention.
Its numbers would be a new experiment and must be reported separately from the
paper's fixed results.

## Run

Download the UCI Online Retail II workbook separately and retain its source
metadata. Then run:

```sh
python main.py --config configs/paper.yaml --data_path path/to/online_retail_II.xlsx
```

The command writes models, metrics, plots, and `outputs/run-manifest.json` under
ignored output directories. The manifest contains the dataset SHA-256,
configuration, source-code hashes, product codes assigned to each split, and
checkpoint hashes.
Run the same command with `--mode eval` to reload those checkpoints and write
`outputs/metrics-eval.json`; it rejects changed inputs, settings, source code,
or weights.
Keep the workbook sheet selection, row counts, dependency versions, hardware, and per-product
predictions with any reported reproduction.

## Published reference values

The paper reports RMSE values of 0.214, 0.198, 0.173, and 0.149 for transfer
learning, multi-task learning, standard MAML, and MetaPrice respectively. It
reports revenue ratios of 0.86, 0.89, 0.93, and 0.97, plus MetaPrice adaptation
latency of 142 ms with standard deviation 18 ms.

These are published reference values. They are not regenerated results in this
repository. A reproduction report must show measured values beside them and
explain any differences without changing the paper.

See [the recorded reproduction run](reproduction-results.md) for a measured
execution of this configuration.

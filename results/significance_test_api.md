# Significance - retrieval `test` / `api`, k=5

Percentile bootstrap over questions, 10000 resamples, seed 0. Mean [95% CI]. Paired comparisons resample the same question ids for A and B; a difference is significant at 5% when its CI excludes 0.

## Confidence intervals per variant

| chunking     | retrieval     |   n | Hit@5                | Recall@5             | precision@5          | MRR                  | nDCG@5               |
|:-------------|:--------------|----:|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| fixed        | bm25          |  52 | 0.788 [0.673, 0.904] | 0.236 [0.167, 0.314] | 0.369 [0.288, 0.450] | 0.680 [0.570, 0.786] | 0.467 [0.378, 0.556] |
| fixed        | dense         |  52 | 0.923 [0.846, 0.981] | 0.329 [0.250, 0.413] | 0.515 [0.431, 0.604] | 0.732 [0.633, 0.825] | 0.595 [0.509, 0.682] |
| fixed        | hybrid        |  52 | 0.942 [0.865, 1.000] | 0.301 [0.227, 0.380] | 0.473 [0.396, 0.554] | 0.740 [0.648, 0.828] | 0.568 [0.487, 0.651] |
| fixed        | hybrid+rerank |  52 | 0.962 [0.904, 1.000] | 0.388 [0.311, 0.469] | 0.638 [0.550, 0.723] | 0.888 [0.815, 0.952] | 0.762 [0.689, 0.830] |
| fixed_header | bm25          |  52 | 0.846 [0.750, 0.942] | 0.267 [0.196, 0.346] | 0.435 [0.354, 0.519] | 0.694 [0.587, 0.797] | 0.514 [0.427, 0.602] |
| fixed_header | dense         |  52 | 0.885 [0.788, 0.962] | 0.319 [0.237, 0.404] | 0.527 [0.442, 0.615] | 0.689 [0.589, 0.788] | 0.573 [0.483, 0.660] |
| fixed_header | hybrid        |  52 | 0.885 [0.788, 0.962] | 0.315 [0.230, 0.405] | 0.473 [0.385, 0.562] | 0.698 [0.608, 0.789] | 0.557 [0.466, 0.648] |
| fixed_header | hybrid+rerank |  52 | 0.942 [0.865, 1.000] | 0.374 [0.291, 0.462] | 0.585 [0.500, 0.669] | 0.859 [0.776, 0.936] | 0.710 [0.629, 0.787] |
| section      | bm25          |  52 | 0.750 [0.635, 0.865] | 0.234 [0.160, 0.315] | 0.377 [0.296, 0.458] | 0.671 [0.559, 0.780] | 0.474 [0.381, 0.567] |
| section      | dense         |  52 | 0.827 [0.712, 0.923] | 0.323 [0.239, 0.411] | 0.458 [0.362, 0.554] | 0.724 [0.620, 0.824] | 0.580 [0.478, 0.678] |
| section      | hybrid        |  52 | 0.904 [0.808, 0.981] | 0.322 [0.241, 0.406] | 0.477 [0.392, 0.562] | 0.700 [0.605, 0.792] | 0.566 [0.479, 0.653] |
| section      | hybrid+rerank |  52 | 0.942 [0.865, 1.000] | 0.385 [0.303, 0.473] | 0.558 [0.473, 0.642] | 0.875 [0.798, 0.942] | 0.712 [0.634, 0.785] |

## Paired comparisons

| comparison (A vs B)                | metric      |   A - B | 95% CI           |      p | significant (5%)   |
|:-----------------------------------|:------------|--------:|:-----------------|-------:|:-------------------|
| rerank vs no rerank (fixed_header) | Hit@5       |   0.058 | [+0.000, +0.135] | 0.0934 | no                 |
| rerank vs no rerank (fixed_header) | precision@5 |   0.112 | [+0.062, +0.169] | 0      | yes                |
| rerank vs no rerank (fixed_header) | MRR         |   0.161 | [+0.081, +0.242] | 0      | yes                |
| hybrid vs BM25 (fixed_header)      | Hit@5       |   0.038 | [-0.058, +0.135] | 0.6014 | no                 |
| hybrid vs BM25 (fixed_header)      | precision@5 |   0.038 | [-0.008, +0.085] | 0.104  | no                 |
| hybrid vs BM25 (fixed_header)      | MRR         |   0.004 | [-0.073, +0.082] | 0.9168 | no                 |
| hybrid vs dense (fixed_header)     | Hit@5       |   0     | [-0.096, +0.096] | 1      | no                 |
| hybrid vs dense (fixed_header)     | precision@5 |  -0.054 | [-0.115, +0.004] | 0.0638 | no                 |
| hybrid vs dense (fixed_header)     | MRR         |   0.009 | [-0.080, +0.098] | 0.838  | no                 |
| fixed vs fixed_header (rerank)     | Hit@5       |   0.019 | [+0.000, +0.058] | 0.7364 | no                 |
| fixed vs fixed_header (rerank)     | precision@5 |   0.054 | [+0.012, +0.096] | 0.006  | yes                |
| fixed vs fixed_header (rerank)     | MRR         |   0.029 | [-0.019, +0.084] | 0.2352 | no                 |
| section vs fixed_header (rerank)   | Hit@5       |   0     | [-0.077, +0.077] | 1      | no                 |
| section vs fixed_header (rerank)   | precision@5 |  -0.027 | [-0.085, +0.035] | 0.3828 | no                 |
| section vs fixed_header (rerank)   | MRR         |   0.016 | [-0.045, +0.077] | 0.5868 | no                 |

## End-to-end: agent vs RAG-only (held-out agent set)

| system   |   n | success rate [95% CI]   |
|:---------|----:|:------------------------|
| agent    |  24 | 0.917 [0.792, 1.000]    |
| RAG-only |  24 | 0.458 [0.250, 0.667]    |

| agent ok / RAG ko | agent ko / RAG ok | McNemar exact p | difference [95% CI] |
|---|---|---|---|
| 11 | 0 | 0.0010 | +0.458 [+0.250, +0.667] |

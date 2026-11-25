# End-to-end scorecard - 24 held-out questions

Chat model `mistralai/mistral-small-2603`, embedder `api`, chunking `fixed_header`.

| system   |   n |   success_rate |   source_accuracy |   tool_selection_acc |   mean_iterations |   mean_tool_calls |   hallucinated_call_rate |   fallback_rate |   final_answer_rate |   tokens_per_q |   cost_usd_per_q |   cost_usd_total |   latency_s_p50 |   latency_s_p95 |   faithfulness |
|:---------|----:|---------------:|------------------:|---------------------:|------------------:|------------------:|-------------------------:|----------------:|--------------------:|---------------:|-----------------:|-----------------:|----------------:|----------------:|---------------:|
| agent    |  24 |         0.9167 |            1      |               0.9583 |            2.6667 |             2.875 |                        0 |               0 |              0.9583 |        4283.92 |           0.0007 |           0.0164 |          2.1105 |          3.9491 |       nan      |
| rag_only |  24 |         0.4583 |            0.7059 |             nan      |          nan      |           nan     |                      nan |             nan |            nan      |        5374.75 |           0.0009 |           0.0225 |          3.021  |          4.1831 |         0.7153 |

## Agent success by question type

| type         |   success |
|:-------------|----------:|
| count        |     1     |
| count+calc   |     1     |
| fr           |     1     |
| fr+calc      |     0     |
| meta         |     1     |
| meta-trap    |     1     |
| multi-hop    |     1     |
| multi-step   |     0.667 |
| rag          |     1     |
| rag+meta     |     1     |
| unanswerable |     1     |

Traces: `results/traces/trace-1795619728.jsonl`

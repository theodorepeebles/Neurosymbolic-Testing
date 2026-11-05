# Classifier comparison results

Run on sft_test.jsonl (101 rows), Colab T4, via classifier_test.ipynb.

- **Exact**: predicted domains == gold domains
- **Covers gold**: no gold domain missing (extras allowed)

## Summary

| Model | Exact | Covers gold | Errors | Avg ms | Avg out tok |
|---|---|---|---|---|---|
| qwen3:8b | 75/101 (74%) | 96/101 (95%) | 0 | — | — |
| qwen3:4b-q4_K_M | 79/101 (78%) | 98/101 (97%) | 1 | 1409 | 52 |
| qwen3:14b | 96/101 (95%) | 98/101 (97%) | 0 | 3879 | 37 |
| qwen3:1.7b | 72/101 (71%) | 73/101 (72%) | 0 | 1424 | 20 |

Notes:
- Plain `qwen3:4b` on Ollama is Qwen3-4B-Thinking-2507, which always thinks (ignores `think: False`) and is far too slow. `qwen3:4b-q4_K_M` is the original hybrid Qwen3 4B, same generation as 8b and 1.7b.
- qwen3:4b-q4_K_M's one error (row 15) was a formatting slip: it separated domains with a tab instead of a comma (`knights_and_knaves\tgrouping`).

## qwen3:8b

```
  [2/101] WRONG  gold=['ordering']  pred=['grouping', 'ordering']
  [7/101] WRONG  gold=['grouping']  pred=['grouping', 'knights_and_knaves']
  [13/101] WRONG  gold=['grouping', 'knights_and_knaves']  pred=['knights_and_knaves', 'ordering']
  [16/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [17/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [20/101] WRONG  gold=['ordering']  pred=['knights_and_knaves', 'ordering']
  [21/101] WRONG  gold=['grouping']  pred=['grouping', 'knights_and_knaves']
  [31/101] WRONG  gold=['ordering']  pred=['knights_and_knaves', 'ordering']
  [32/101] WRONG  gold=['grouping', 'knights_and_knaves']  pred=['grouping', 'knights_and_knaves', 'ordering']
  [34/101] WRONG  gold=['grouping']  pred=['grouping', 'ordering']
  [41/101] WRONG  gold=['ordering']  pred=['knights_and_knaves', 'ordering']
  [42/101] WRONG  gold=['ordering']  pred=['grouping', 'ordering']
  [43/101] WRONG  gold=['grouping', 'knights_and_knaves']  pred=['grouping', 'knights_and_knaves', 'ordering']
  [46/101] WRONG  gold=['grouping', 'knights_and_knaves']  pred=['grouping', 'knights_and_knaves', 'ordering']
  [51/101] WRONG  gold=['grouping', 'ordering']  pred=['grouping', 'knights_and_knaves', 'ordering']
  [52/101] WRONG  gold=['grouping', 'ordering']  pred=['grouping', 'knights_and_knaves', 'ordering']
  [53/101] WRONG  gold=['grouping', 'knights_and_knaves']  pred=['grouping', 'knights_and_knaves', 'ordering']
  [54/101] WRONG  gold=['grouping']  pred=['grouping', 'knights_and_knaves']
  [58/101] WRONG  gold=['grouping']  pred=['grouping', 'knights_and_knaves']
  [69/101] WRONG  gold=['grouping', 'ordering']  pred=['grouping', 'knights_and_knaves', 'ordering']
  [77/101] WRONG  gold=['ordering']  pred=['knights_and_knaves', 'ordering']
  [79/101] WRONG  gold=['knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves', 'ordering']
  [80/101] WRONG  gold=['grouping', 'knights_and_knaves']  pred=['knights_and_knaves', 'ordering']
  [81/101] WRONG  gold=['grouping']  pred=['grouping', 'ordering']
  [86/101] WRONG  gold=['grouping']  pred=['grouping', 'ordering']
  [97/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  exact 75/101  covers 96/101  errors 0
```

## qwen3:4b-q4_K_M

```
  [3/101] WRONG  gold=['ordering']  pred=['grouping', 'ordering']
  [7/101] WRONG  gold=['grouping']  pred=['grouping', 'ordering']
  [15/101] ERROR  gold=['knights_and_knaves', 'ordering']  Classifier returned unknown domains: ['knights_and_knaves\tgrouping']
  [16/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [20/101] WRONG  gold=['ordering']  pred=['grouping', 'ordering']
  [21/101] WRONG  gold=['grouping']  pred=['grouping', 'ordering']
  [28/101] WRONG  gold=['knights_and_knaves']  pred=['grouping', 'knights_and_knaves']
  [34/101] WRONG  gold=['grouping']  pred=['grouping', 'ordering']
  [41/101] WRONG  gold=['ordering']  pred=['grouping', 'ordering']
  [42/101] WRONG  gold=['ordering']  pred=['grouping', 'ordering']
  [45/101] WRONG  gold=['grouping', 'ordering']  pred=['grouping', 'knights_and_knaves', 'ordering']
  [49/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [54/101] WRONG  gold=['grouping']  pred=['grouping', 'ordering']
  [55/101] WRONG  gold=['grouping']  pred=['grouping', 'ordering']
  [58/101] WRONG  gold=['grouping']  pred=['grouping', 'ordering']
  [60/101] WRONG  gold=['ordering']  pred=['grouping', 'ordering']
  [77/101] WRONG  gold=['ordering']  pred=['grouping', 'ordering']
  [81/101] WRONG  gold=['grouping']  pred=['grouping', 'ordering']
  [86/101] WRONG  gold=['grouping']  pred=['grouping', 'ordering']
  [88/101] WRONG  gold=['knights_and_knaves']  pred=['knights_and_knaves', 'ordering']
  [90/101] WRONG  gold=['grouping']  pred=['grouping', 'ordering']
  [95/101] WRONG  gold=['ordering']  pred=['grouping', 'ordering']
  exact 79/101  covers 98/101  errors 1
```

## qwen3:14b

```
  [12/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [16/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [30/101] WRONG  gold=['grouping', 'knights_and_knaves']  pred=['grouping', 'knights_and_knaves', 'ordering']
  [65/101] WRONG  gold=['knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves', 'ordering']
  [85/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  exact 96/101  covers 98/101  errors 0
```

## qwen3:1.7b

```
  [5/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [8/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [9/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [10/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [12/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [14/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [17/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [19/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [23/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [26/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [27/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [33/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [38/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [39/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [40/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [49/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [50/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [53/101] WRONG  gold=['grouping', 'knights_and_knaves']  pred=['grouping']
  [57/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [62/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [64/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [70/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [71/101] WRONG  gold=['knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves', 'ordering']
  [72/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [73/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [83/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [91/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['grouping', 'knights_and_knaves']
  [99/101] WRONG  gold=['grouping', 'knights_and_knaves', 'ordering']  pred=['knights_and_knaves', 'ordering']
  [100/101] WRONG  gold=['grouping', 'ordering']  pred=['grouping']
  exact 72/101  covers 73/101  errors 0
```

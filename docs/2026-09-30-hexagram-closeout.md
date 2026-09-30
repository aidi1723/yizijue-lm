# YiZiJue-LM hexagram closeout

Date: 2026-09-30

The hexagram linear head is closed on the frozen Qwen3-0.6B base and v5 LoRA.
Served temperature stays 1.0. Weights were not replaced after the temperature sweep.

## Gates

| Set | Count | State accuracy | Halt recall | Unsafe allow |
| --- | ---: | ---: | ---: | ---: |
| Real test | 300 | 0.9833 | 0.9861 | 0 |
| Coverage test | 480 | 1.0 | no halt gold | 0 |
| original-212 | 212 | 0.9717 | 0.9938 | 0 |

Coverage macro recall and minimum recall are both 1.0.

The moving head reads explicit line values 6, 7, 8, and 9. Held-out line accuracy is 0.9948. Changed hexagrams stay inside the 64. The action head was not replaced.

## Closed without another training run

- Ordinary prose has no 6/7/8/9 gold.
- A review may only name a hexagram. No second model proposes a recast.
- original-212 rows where the gold label conflicts with the halt rule are not relabeled. Of 27 verifier-gold rows, 21 dangerous sentences stay on the keyword halt, and 6 plain pytest sentences stay on the kan hexagram.

Machine-readable numbers: `docs/2026-09-30-training-closeout.json`.

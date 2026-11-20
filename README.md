# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wizzy-bounty/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wizzy-bounty/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecast of every Grand Theft Auto VI PS5 trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95) per trophy, scored by mean
pinball loss on the logit scale.

How it works, per trophy, on the logit scale:

- The case's reads are compressed into distinct page snapshots — the page only changes
  when Exophase refreshes it, so duplicate reads carry no new information.
- With a rich history, a recency-weighted least-squares slope of the logit rate is fit
  on two clocks (ln(1 + hours since release) and plain hours) and the two extrapolations
  averaged; slopes are shrunk toward the pooled slope, the forecast is anchored at the
  last observed rate, and the total move is capped. The spread combines the observed
  snapshot-to-snapshot volatility over the horizon, the slope's standard error, a term
  proportional to the extrapolated move, and a floor.
- With a launch-frozen history (the page never changed, the tracked base is still the
  few-hundred-player pre-release population, and the case sits within two days of
  release), the first refresh inside the horizon folds in the launch surge: the centre
  shifts by ln(1.34 / 6.0) in odds — earners grow modestly while the tracked base
  multiplies — with a spread sized for the surge's uncertainty.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` case file in the input directory and writes one `<id>.out`
forecast per case. Python 3.13, standard library only, no network, deterministic:
two runs of a case produce byte-identical output.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

## Model notes

- **Model:** each trophy's earn rate is projected with a recency-weighted slope on the logit scale, estimated on two clocks and shrunk toward the pooled cross-trophy slope; the forecast anchors at the last observed rate and caps the total move at ±2,8 logit.
- **Spread:** predictive intervals come from a snapshot-volatility random walk accumulated over the forecast horizon, plus the slope's standard error, plus 0,30×|move| so bigger projected moves carry wider bands.
- **Validation:** local practice reconstruction losses land around 0,08–0,19 versus the house baseline at ≈1,51, and simulated week-long cuts score 0,154 against 0,241 for the curve baseline.
- **Runtime:** about 0,06 s per case, Python standard library only, fully deterministic.

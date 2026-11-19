# GTA VI Trophy Rarity Forecast

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Whydiriki700/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Whydiriki700/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01):
forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, the earn
rate Exophase will report for it at a future moment — the percentage of
tracked players who have earned that trophy — as five quantiles
(5%, 25%, 50%, 75%, 95%) per trophy.

## How it works

Each case hands the program the full history of trophy-page reads taken
before a checkpoint, and asks for the rates at a later target time. Every
trophy is forecast on its own, from two complementary angles that are then
averaged on the logit scale:

- **Structural path.** A rate is earners divided by tracked players, and
  the two counts move for different reasons. The program projects each
  count separately — a recency-weighted straight line through the log of
  earners and the log of players, with slopes capped so nothing explodes —
  and divides the projections. This is what lets the forecast fall when
  the player pool is swelling faster than the earner pool, the usual
  launch-week dilution pattern.
- **Direct path.** A recency-weighted regression of the rate's logit
  against ln(1 + hours since the series began), re-anchored at the last
  fitted point. Its slope is shrunk toward zero in proportion to how
  little data backs it, so a trophy seen twice stays near its last rate
  instead of extrapolating wildly.

The spread around the median blends the direct fit's residuals with a
random-walk term grown from the read-to-read logit steps over the horizon,
is floored at half the square root of the horizon in days, and is capped —
long horizons get honestly wide intervals, never absurd ones. A trophy
with no usable reads falls back to a low neutral rate with a wide spread,
so every case always produces a complete, valid answer. Rates are clamped
to [0.01, 99.99] before any logit is taken, outputs are rounded to four
decimals with quantiles kept ascending, and the program is fully
deterministic: the same case always yields the same bytes.

## Running it

The program is a single standard-library Python file (Python 3.13),
`main.py` at the repository root. It reads every `*.in` case file from
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `<id>.out` JSON
file per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`):

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each output file holds exactly
`{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}...]}` — one row per
trophy, in id order, every value within 0–100.

New to make.fun? Sign up through my referral link:
https://make.fun/r/vwpycztt

## License

MIT — see `LICENSE`.

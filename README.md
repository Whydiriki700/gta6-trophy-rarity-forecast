# Forecasting GTA VI Trophy Rarity

[![Bounty badge](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![Rank badge](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Whydiriki700/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![Reward badge](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Whydiriki700/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

This repository is my entry to the make.fun competition
gta6-trophy-rarity-forecast-01. The question it answers is simple to
state and awkward to forecast: for each of the 55 PlayStation 5
trophies of Grand Theft Auto VI, what percentage of the players tracked
by Exophase will have earned that trophy at the target time given in a
case?

## What the program does

`main.py` reads every case file (`<id>.in`) in the input directory and
writes a matching `<id>.out` file. Each output row holds five quantiles
(at the 0.05, 0.25, 0.5, 0.75 and 0.95 levels) for one trophy, in trophy
id order.

The forecast for a trophy blends two projections in logit space:

1. A structural projection. Earners are recovered from each read as
   rate times players tracked. Earners and players are each fitted in
   log space against time with recency-weighted least squares and
   projected forward with guarded slopes. Dividing the projected
   earners by the projected players gives a rate that already contains
   the dilution caused by a growing tracked pool.
2. A direct projection. The logit of the observed rate is fitted
   against ln(1 + hours) with the same recency weighting. The fitted
   slope is shrunk toward zero for short series or series with few
   distinct values, and the projection is anchored at the last fitted
   point rather than at a single noisy read.

The median is the average of the two projections when the structural
one has enough data, and the direct projection on its own otherwise.
The interval width combines the direct fit's residuals with the
observed step-to-step logit volatility rescaled to the horizon. That
width is floored at 0.5 * sqrt(horizon in days), so it is never
narrower than a plain random-walk baseline, and capped at 2.5. Trophies
with no usable reads fall back to the case-wide median rate with a wide
band, and a case with no history at all gets a low flat forecast.
Nothing in the program crashes on null rates or very short series.

## Running it

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 main.py
```

When the two environment variables are absent the program defaults to
`/input` and `/output`. It needs Python 3.13 and the standard library
only, uses no network, and is deterministic: repeated runs on the same
input produce byte-identical files.

## Referral

If you want to try make.fun competitions yourself, here is the
referral link: https://make.fun/r/vwpycztt

## Licence

Released under the MIT licence; see the LICENSE file in this
repository.

# GTA VI Trophy Rarity Forecaster

[![Bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Whydiriki700/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Whydiriki700/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for the make.fun **gta6-trophy-rarity-forecast-01** bounty (USDC prize
pool). The program forecasts, for every one of GTA VI's 55 PS5 trophies, the
Exophase earn rate at the case target time, as five quantiles per trophy.

## How it works

For each trophy the forecaster combines two paths over the case's read
history:

- **Structural path.** Earners (rate x tracked players) and the tracked
  player pool are projected separately in log space, so the dilution that
  drags every rate down when the pool grows is modelled directly instead of
  being extrapolated as if it were player behaviour.
- **Direct path.** A recency-weighted least-squares trend of the rate in
  logit space against log time, with the slope shrunk toward zero when the
  series is short or flat.

The median is the average of the two paths in logit space. The spread comes
from the trend fit's residuals plus the observed read-to-read logit
volatility scaled by the horizon, floored at a random-walk spread so quiet
trophies keep honest, wide intervals.

Pure Python 3 standard library, no network, deterministic: two runs on the
same input produce byte-identical output.

## Run

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Reads every `<id>.in` case in the input directory and writes `<id>.out`
files in the output directory, in the case format the bounty specifies.

## Join make.fun

Agent bounties like this one are open to join:
[make.fun/r/vwpycztt](https://make.fun/r/vwpycztt)

## License

MIT, see LICENSE.

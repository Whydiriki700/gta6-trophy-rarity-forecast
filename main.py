#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster.

For every trophy in a case, forecast its Exophase earn rate (percent of
tracked players who earned it) at the case target time, as five quantiles.

Method, per trophy, from the case history of page reads:
  * a structural path: earners (= rate x players / 100) and tracked players
    are projected separately in log space, so the dilution that hits every
    rate when the tracked pool grows is modelled instead of extrapolated;
  * a direct path: a recency-weighted least-squares trend of the rate in
    logit space against log time, with the slope shrunk toward zero when
    the series is short or flat;
  * the median is the average of the two paths in logit space;
  * the spread comes from the trend fit's residuals plus the observed
    read-to-read logit volatility scaled by the horizon, floored at the
    house baseline's random-walk spread so quiet trophies stay honest.

Standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
LO, HI = 0.01, 99.99


def parse_ts(text):
    # Case timestamps are second-precision; history reads carry fractional
    # seconds. fromisoformat handles both once the Z suffix is expanded.
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def clamp_rate(r):
    return max(LO, min(HI, r))


def logit_pct(r):
    p = clamp_rate(r) / 100.0
    return math.log(p / (1.0 - p))


def pct_from_logit(x):
    return 100.0 / (1.0 + math.exp(-x))


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns (a, b, resid_sd)."""
    n = len(xs)
    sw = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for r, w in zip(resid, ws)) / sw
    return a, b, math.sqrt(max(var, 0.0))


def recency_weights(n):
    if n <= 1:
        return [1.0] * n
    return [0.35 + 0.65 * (k / (n - 1)) for k in range(n)]


def project_players(hours, players, target_h):
    """Project tracked players to the target hour, log-linear, guarded."""
    pts = [(h, math.log(max(p, 1.0))) for h, p in zip(hours, players) if p > 0]
    last = players[-1]
    if len(pts) < 2:
        return float(last)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    a, b, _ = wls(xs, ys, recency_weights(len(xs)))
    b = max(0.0, min(0.02, b))
    proj = math.exp(a + b * target_h)
    return max(last * 0.9, min(last * 20.0, proj))


def forecast_trophy(series, target_h, horizon_days, players_proj):
    """series: list of (hour, rate, players). Returns five quantiles."""
    last_rate = series[-1][1]
    hours = [s[0] for s in series]
    rates = [s[1] for s in series]
    n = len(series)

    # Direct path: logit trend against log time.
    xs = [math.log1p(h) for h in hours]
    ys = [logit_pct(r) for r in rates]
    a, b, resid_sd = wls(xs, ys, recency_weights(n))
    distinct = len({round(r, 4) for r in rates})
    shrink = min(1.0, n / 8.0) * (1.0 if distinct >= 3 else 0.35)
    if n < 3:
        shrink = 0.0
    # Anchor at the last fitted point; apply the shrunk slope only over the
    # remaining log-time distance to the target.
    fitted_last = a + b * xs[-1]
    med_logit_direct = fitted_last + b * shrink * (math.log1p(target_h) - xs[-1])

    # Structural path: earners and players projected separately.
    earners = [(r / 100.0) * p for _, r, p in series]
    med_rate_struct = None
    epts = [(h, math.log(e)) for h, e in zip(hours, earners) if e > 0]
    if len(epts) >= 2 and players_proj > 0:
        exs = [p[0] for p in epts]
        eys = [p[1] for p in epts]
        ea, eb, _ = wls(exs, eys, recency_weights(len(exs)))
        eb = max(-0.005, min(0.03, eb))
        e_proj = math.exp(ea + eb * target_h)
        e_last = earners[-1]
        e_proj = max(e_last * 0.98, min(e_last + players_proj, e_proj))
        med_rate_struct = clamp_rate(100.0 * e_proj / players_proj)

    if med_rate_struct is not None:
        med_logit = 0.5 * med_logit_direct + 0.5 * logit_pct(med_rate_struct)
    else:
        med_logit = med_logit_direct

    # Spread: fit residuals plus step volatility over the horizon, floored
    # at the house baseline's 0.5 * sqrt(days) random-walk spread.
    steps = []
    for k in range(1, n):
        dh = hours[k] - hours[k - 1]
        if dh > 0:
            steps.append((ys[k] - ys[k - 1]) / math.sqrt(dh))
    if len(steps) >= 2:
        m = sum(steps) / len(steps)
        step_sd = math.sqrt(sum((s - m) ** 2 for s in steps) / len(steps))
    elif steps:
        step_sd = abs(steps[0])
    else:
        step_sd = 0.0
    horizon_hours = max(target_h - hours[-1], 0.0)
    sigma = math.sqrt(resid_sd**2 + (step_sd * math.sqrt(horizon_hours)) ** 2)
    sigma = max(sigma, 0.5 * math.sqrt(max(horizon_days, 0.0)))
    sigma = min(sigma, 2.5)

    qs = [pct_from_logit(med_logit + z * sigma) for z in Z]
    qs = [round(max(0.0, min(100.0, q)), 4) for q in qs]
    for k in range(1, len(qs)):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return qs


def run_case(case):
    trophies = case["trophies"]
    history = case["history"]
    target_h_epoch = parse_ts(case["target"])
    rows_out = []
    if not history:
        for t in trophies:
            rows_out.append({"id": t["id"], "q": [0.05, 0.2, 0.5, 1.5, 4.0]})
        return {"trophies": rows_out}

    t0 = parse_ts(history[0]["at"])
    hours = [(parse_ts(r["at"]) - t0) / 3600.0 for r in history]
    target_h = (target_h_epoch - t0) / 3600.0
    horizon_days = case["horizon_minutes"] / 1440.0
    players = [r["players_tracked"] for r in history]
    players_proj = project_players(hours, players, target_h)

    last_read = history[-1]["rates"]
    fallback_pool = sorted(
        r for r in last_read if r is not None
    )
    fallback_rate = (
        fallback_pool[len(fallback_pool) // 2] if fallback_pool else 0.5
    )

    for idx, t in enumerate(trophies):
        series = []
        for k, read in enumerate(history):
            r = read["rates"][idx] if idx < len(read["rates"]) else None
            if r is not None:
                series.append((hours[k], float(r), float(players[k])))
        if not series:
            base = clamp_rate(fallback_rate)
            qs = [round(base * f, 4) for f in (0.4, 0.7, 1.0, 1.4, 2.0)]
            qs = [max(0.0, min(100.0, q)) for q in qs]
            rows_out.append({"id": t["id"], "q": qs})
            continue
        qs = forecast_trophy(series, target_h, horizon_days, players_proj)
        rows_out.append({"id": t["id"], "q": qs})
    return {"trophies": rows_out}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = run_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

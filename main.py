#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates for make.fun.

For every case file in the input directory this program projects, trophy by
trophy, the distribution of the trophy's Exophase earn rate (percent of
tracked players who have earned it) at the case's target time, and writes
five quantiles per trophy to the output directory.

Two independent projections are blended for each trophy:

* a structural projection, which extrapolates the earner count and the
  tracked-player count separately (in log space) and divides them, so it
  captures the dilution that pushes rates down when the player pool grows
  faster than the earner pool;
* a direct projection, a recency-weighted trend of the rate's logit against
  log time, with its slope shrunk toward zero so short or flat series stay
  close to their last observed level.

The reported spread combines the direct fit's residuals with a random-walk
term built from read-to-read logit steps, floored by a horizon term and
capped, so longer horizons honestly widen the quantiles.

Standard library only, no network, no randomness: repeated runs on the same
case produce byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# z-values for the five forecast levels 0.05, 0.25, 0.5, 0.75, 0.95.
Z_LEVELS = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)

RATE_FLOOR = 0.01          # rates are clamped into [RATE_FLOOR, RATE_CEIL]
RATE_CEIL = 99.99          # before any logit is taken
SIGMA_CAP = 2.5            # widest logit spread ever reported
LOG_SLOPE_LIMIT = 0.01     # per-hour cap on log-space growth slopes
DIRECT_SLOPE_LIMIT = 2.0   # cap on the direct logit-vs-log-time slope
DIRECT_TRAVEL_LIMIT = 3.0  # cap on the direct path's total logit travel
                           # from its anchor, however long the horizon

FALLBACK_RATE = 1.0        # neutral low rate for a trophy with no data
FALLBACK_SIGMA = 2.0       # ... with a wide spread


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def clamp_rate(rate):
    return min(max(rate, RATE_FLOOR), RATE_CEIL)


def logit_of_percent(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def percent_of_logit(value):
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def parse_timestamp(text):
    """Epoch seconds for an ISO-8601 timestamp, or None if unparseable.

    Handles a trailing 'Z' and fractional seconds, which the case files mix
    freely between second-precision and sub-second forms.
    """
    if not isinstance(text, str) or not text:
        return None
    cleaned = text.strip()
    if cleaned.endswith(("Z", "z")):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def weighted_line_fit(xs, ys, ws):
    """Weighted least squares y ~= a + b*x.

    Returns (a, b, fitted_values).  When every x is the same the slope is 0
    and the intercept is the weighted mean.
    """
    total_w = sum(ws)
    if total_w <= 0:
        return 0.0, 0.0, [0.0] * len(xs)
    mean_x = sum(w * x for x, w in zip(xs, ws)) / total_w
    mean_y = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return mean_y, 0.0, [mean_y] * len(xs)
    sxy = sum(w * (x - mean_x) * (y - mean_y) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    return intercept, slope, [intercept + slope * x for x in xs]


def recency_weights(times):
    """Exponential weights: the newest observation weighs 1, older ones decay
    with a half-life of a third of the observed span (at least an hour)."""
    if not times:
        return []
    newest = max(times)
    span = newest - min(times)
    half_life = max(span / 3.0, 1.0)
    return [0.5 ** ((newest - t) / half_life) for t in times]


# --------------------------------------------------------------------------
# per-trophy forecast
# --------------------------------------------------------------------------

def structural_median_logit(obs, target_hour):
    """Project earners and players separately in log space and divide.

    obs is a list of (hour, rate, players).  Returns a median logit, or None
    when the path has too little to work with.
    """
    usable = [(h, r, p) for h, r, p in obs if p and p > 0]
    if len(usable) < 2:
        return None
    hours = [h for h, _, _ in usable]
    if max(hours) - min(hours) <= 0:
        return None
    weights = recency_weights(hours)

    log_players = [math.log(p) for _, _, p in usable]
    log_earners = [math.log(max(r * p / 100.0, 1e-9)) for _, r, p in usable]

    a_p, b_p, _ = weighted_line_fit(hours, log_players, weights)
    a_e, b_e, _ = weighted_line_fit(hours, log_earners, weights)
    b_p = min(max(b_p, -LOG_SLOPE_LIMIT), LOG_SLOPE_LIMIT)
    b_e = min(max(b_e, -LOG_SLOPE_LIMIT), LOG_SLOPE_LIMIT)

    players_hat = math.exp(a_p + b_p * target_hour)
    earners_hat = math.exp(a_e + b_e * target_hour)
    if players_hat <= 0:
        return None
    earners_hat = min(max(earners_hat, 0.0), players_hat)
    return logit_of_percent(100.0 * earners_hat / players_hat)


def direct_fit(obs):
    """Recency-weighted fit of logit(rate) against ln(1 + hours).

    Returns (last_x, fitted_at_last, shrunk_slope, residual_sigma,
    step_sigma, median_step_hours) or None when there are no observations.
    """
    if not obs:
        return None
    hours = [h for h, _, _ in obs]
    xs = [math.log1p(h) for h in hours]
    ys = [logit_of_percent(r) for _, r, _ in obs]
    weights = recency_weights(hours)

    intercept, slope, fitted = weighted_line_fit(xs, ys, weights)
    n = len(obs)
    shrink = n / (n + 4.0)          # short series trust their slope less
    slope = min(max(slope * shrink, -DIRECT_SLOPE_LIMIT), DIRECT_SLOPE_LIMIT)

    residual_sigma = None
    if n >= 3:
        total_w = sum(weights)
        var = sum(w * (y - f) ** 2 for y, f, w in zip(ys, fitted, weights))
        residual_sigma = math.sqrt(var / total_w)

    step_sigma = None
    median_step = None
    if n >= 3:
        diffs = [ys[i + 1] - ys[i] for i in range(n - 1)]
        mean_d = sum(diffs) / len(diffs)
        step_sigma = math.sqrt(
            sum((d - mean_d) ** 2 for d in diffs) / len(diffs))
        gaps = sorted(hours[i + 1] - hours[i] for i in range(n - 1))
        median_step = gaps[len(gaps) // 2]
        if median_step <= 0:
            median_step = None
    return xs[-1], fitted[-1], slope, residual_sigma, step_sigma, median_step


def forecast_trophy(obs, target_hour, horizon_hours):
    """(median_logit, sigma) for one trophy.

    obs holds (hour, rate, players) triples in time order, hours measured
    from the trophy's first observation.
    """
    horizon_days = max(horizon_hours, 0.0) / 24.0
    sigma_floor = 0.5 * math.sqrt(horizon_days)

    if not obs:
        return logit_of_percent(FALLBACK_RATE), min(
            max(FALLBACK_SIGMA, sigma_floor), SIGMA_CAP)

    fit = direct_fit(obs)
    last_x, fitted_last, slope, resid, step, median_step = fit
    target_x = math.log1p(max(target_hour, 0.0))
    travel = slope * (target_x - last_x)
    travel = min(max(travel, -DIRECT_TRAVEL_LIMIT), DIRECT_TRAVEL_LIMIT)
    direct_logit = fitted_last + travel

    structural = structural_median_logit(obs, target_hour)
    if structural is not None:
        median_logit = 0.5 * (direct_logit + structural)
    else:
        median_logit = direct_logit

    parts = []
    if resid is not None:
        parts.append(resid ** 2)
    if step is not None and median_step:
        walk = step * math.sqrt(max(horizon_hours, 0.0) / median_step)
        parts.append(walk ** 2)
    if parts:
        sigma = math.sqrt(sum(parts))
    else:
        sigma = max(sigma_floor, 0.6)
    sigma = min(max(sigma, sigma_floor), SIGMA_CAP)
    return median_logit, sigma


def quantile_row(median_logit, sigma):
    values = []
    for z in Z_LEVELS:
        q = percent_of_logit(median_logit + z * sigma)
        values.append(round(min(max(q, 0.0), 100.0), 4))
    for i in range(1, len(values)):     # enforce ascending after rounding
        if values[i] < values[i - 1]:
            values[i] = values[i - 1]
    return values


# --------------------------------------------------------------------------
# case handling
# --------------------------------------------------------------------------

def collect_observations(case):
    """Per-trophy observation lists plus the forecast timing.

    Returns (per_trophy, target_ts, horizon_hours) where per_trophy maps a
    trophy's index in the case's trophy list to (observations, base_ts);
    each observation is (hours since that trophy's first observation, rate,
    players).
    """
    history = case.get("history") or []
    reads = []
    last_ts = None
    for read in history:
        if not isinstance(read, dict):
            continue
        ts = parse_timestamp(read.get("at"))
        if ts is None:
            ts = last_ts if last_ts is not None else 0.0
        last_ts = ts
        reads.append((ts, read))

    at_ts = parse_timestamp(case.get("at"))
    target_ts = parse_timestamp(case.get("target"))
    if target_ts is None and at_ts is not None:
        target_ts = at_ts + 60.0 * float(case.get("horizon_minutes") or 0)
    if at_ts is not None and target_ts is not None:
        horizon_hours = max((target_ts - at_ts) / 3600.0, 0.0)
    elif case.get("horizon_minutes") is not None:
        horizon_hours = max(float(case["horizon_minutes"]) / 60.0, 0.0)
    else:
        horizon_hours = 0.0

    trophies = case.get("trophies") or []
    per_trophy = {}
    for index in range(len(trophies)):
        obs = []
        base_ts = None
        for ts, read in reads:
            rates = read.get("rates") or []
            if index >= len(rates):
                continue
            rate = rates[index]
            if rate is None:
                continue
            try:
                rate = float(rate)
            except (TypeError, ValueError):
                continue
            if base_ts is None:
                base_ts = ts
            players = read.get("players_tracked")
            try:
                players = float(players) if players is not None else None
            except (TypeError, ValueError):
                players = None
            obs.append(((ts - base_ts) / 3600.0, rate, players))
        # The fits, the step diffs and the last-point anchor all assume
        # chronological order, and two reads can share a timestamp; sort
        # and keep only the later read of any duplicated hour.
        obs.sort(key=lambda point: point[0])
        deduped = []
        for point in obs:
            if deduped and deduped[-1][0] == point[0]:
                deduped[-1] = point
            else:
                deduped.append(point)
        per_trophy[index] = (deduped, base_ts)
    return per_trophy, target_ts, horizon_hours


def forecast_case(case):
    trophies = case.get("trophies") or []
    per_trophy, target_ts, horizon_hours = collect_observations(case)
    rows = []
    for index, trophy in enumerate(trophies):
        trophy_id = trophy.get("id", index + 1) if isinstance(trophy, dict) \
            else index + 1
        obs, base_ts = per_trophy.get(index, ([], None))
        if base_ts is not None and target_ts is not None:
            target_hour = (target_ts - base_ts) / 3600.0
        else:
            target_hour = horizon_hours
        median_logit, sigma = forecast_trophy(obs, target_hour, horizon_hours)
        rows.append({"id": trophy_id, "q": quantile_row(median_logit, sigma)})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def fallback_output(case):
    """A valid-shaped output built from bare trophy ids, used if anything
    about a case's data defeats the real forecast."""
    rows = []
    for index, trophy in enumerate(case.get("trophies") or []):
        trophy_id = trophy.get("id", index + 1) if isinstance(trophy, dict) \
            else index + 1
        median_logit, sigma = forecast_trophy([], 0.0, 0.0)
        rows.append({"id": trophy_id, "q": quantile_row(median_logit, sigma)})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    try:
        names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    except OSError:
        return 0
    for name in names:
        case = None
        try:
            with open(os.path.join(input_dir, name), "r",
                      encoding="utf-8") as handle:
                case = json.load(handle)
            result = forecast_case(case)
        except Exception:
            if not isinstance(case, dict):
                continue        # no trophy ids to key a fallback on
            result = fallback_output(case)
        out_name = name[:-len(".in")] + ".out"
        with open(os.path.join(output_dir, out_name), "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

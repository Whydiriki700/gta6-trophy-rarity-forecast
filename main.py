#!/usr/bin/env python3
"""Trophy rarity forecaster for GTA VI (Exophase earn rates).

For every trophy in a case this program projects the earn rate at the
case target time and reports five quantiles of that rate.  Two views of
the same history are blended:

* a structural view that tracks earners and tracked players separately,
  so growth of the tracked pool dilutes the rate in a mechanical way;
* a direct view that follows the logit of the rate against log time.

The spread around the blended median is never allowed to be tighter
than a simple random-walk baseline for the same horizon.  Only the
Python standard library is used and the output is deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z_VALUES = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
RATE_LO = 0.01
RATE_HI = 99.99


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp, fractional seconds optional."""
    if text is None:
        return None
    cleaned = str(text).strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def clamp_rate(rate):
    if rate < RATE_LO:
        return RATE_LO
    if rate > RATE_HI:
        return RATE_HI
    return rate


def logit_of_percent(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def percent_from_logit(value):
    # Numerically tame sigmoid, returned on the percent scale.
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return 100.0 * exp_value / (1.0 + exp_value)


def recency_weights(hours, half_life=24.0):
    """Exponentially decaying weights; the newest read weighs 1."""
    if not hours:
        return []
    newest = max(hours)
    weights = []
    for h in hours:
        age = newest - h
        weights.append(0.5 ** (age / half_life))
    return weights


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~= a + b*x.  Returns (a, b)."""
    total_w = sum(ws)
    if total_w <= 0.0:
        return (ys[0] if ys else 0.0), 0.0
    mean_x = sum(w * x for x, w in zip(xs, ws)) / total_w
    mean_y = sum(w * y for y, w in zip(ys, ws)) / total_w
    sxx = 0.0
    sxy = 0.0
    for x, y, w in zip(xs, ys, ws):
        dx = x - mean_x
        sxx += w * dx * dx
        sxy += w * dx * (y - mean_y)
    if sxx < 1e-12:
        return mean_y, 0.0
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


def residual_sd(xs, ys, intercept, slope):
    n = len(xs)
    if n < 3:
        return 0.0
    acc = 0.0
    for x, y in zip(xs, ys):
        diff = y - (intercept + slope * x)
        acc += diff * diff
    return math.sqrt(acc / (n - 2))


def direct_path(hours, logits, hours_target):
    """Logit trend against ln(1 + hours), anchored at the last fit.

    Returns (projected logit, residual sd, step volatility per sq-hour)
    or None when there is nothing to fit.
    """
    n = len(logits)
    if n == 0:
        return None
    if n == 1:
        return logits[0], 0.0, 0.0

    xs = [math.log1p(h) for h in hours]
    ws = recency_weights(hours)
    intercept, slope = weighted_line(xs, logits, ws)

    distinct = len({round(v, 6) for v in logits})
    if distinct < 2:
        shrink = 0.0
    else:
        shrink = ((n - 2.0) / (n + 2.0)) * ((distinct - 1.0) / distinct)
        if shrink < 0.0:
            shrink = 0.0
        if shrink > 1.0:
            shrink = 1.0
    slope = slope * shrink
    # Re-anchor: with the shrunk slope, pass the line through the
    # weighted centre so the last fitted point carries the forecast.
    total_w = sum(ws)
    mean_x = sum(w * x for x, w in zip(xs, ws)) / total_w
    mean_y = sum(w * y for y, w in zip(logits, ws)) / total_w
    intercept = mean_y - slope * mean_x

    if slope > 2.5:
        slope = 2.5
    elif slope < -2.5:
        slope = -2.5

    x_last = xs[-1]
    x_target = math.log1p(max(hours_target, hours[-1]))
    delta = slope * (x_target - x_last)
    if delta > 3.0:
        delta = 3.0
    elif delta < -3.0:
        delta = -3.0
    fitted_last = intercept + slope * x_last
    projected = fitted_last + delta

    sd = residual_sd(xs, logits, intercept, slope)

    # Read-to-read step volatility, normalised per sqrt hour so it can
    # be rescaled to any horizon as a random walk would be.
    norm_steps = []
    for i in range(1, n):
        dt = hours[i] - hours[i - 1]
        if dt > 1e-9:
            norm_steps.append((logits[i] - logits[i - 1]) / math.sqrt(dt))
    if norm_steps:
        step_vol = math.sqrt(
            sum(v * v for v in norm_steps) / len(norm_steps)
        )
    else:
        step_vol = 0.0
    return projected, sd, step_vol


def structural_path(hours, rates, players_list, hours_target):
    """Project earners and players separately, then divide.

    Returns a projected logit, or None when this view has too little
    usable data (fewer than three reads carrying both fields).
    """
    usable = [
        (h, r, p)
        for h, r, p in zip(hours, rates, players_list)
        if r is not None and p is not None and p > 0
    ]
    if len(usable) < 3:
        return None
    hs = [u[0] for u in usable]
    log_earners = []
    log_players = []
    for _, rate, players in usable:
        earners = clamp_rate(rate) / 100.0 * players
        if earners < 1e-6:
            earners = 1e-6
        log_earners.append(math.log(earners))
        log_players.append(math.log(players))

    ws = recency_weights(hs)

    def project(log_values):
        intercept, slope = weighted_line(hs, log_values, ws)
        if slope > 0.2:
            slope = 0.2
        elif slope < -0.2:
            slope = -0.2
        fitted_last = intercept + slope * hs[-1]
        change = slope * (max(hours_target, hs[-1]) - hs[-1])
        if change > 2.0:
            change = 2.0
        elif change < -2.0:
            change = -2.0
        return fitted_last + change

    earners_proj = project(log_earners)
    players_proj = project(log_players)
    rate_proj = 100.0 * math.exp(earners_proj - players_proj)
    return logit_of_percent(rate_proj)


def quantiles_from(median_logit, sigma):
    values = []
    for z in Z_VALUES:
        pct = percent_from_logit(median_logit + z * sigma)
        rounded = round(pct, 4)
        if rounded < 0.0:
            rounded = 0.0
        if rounded > 100.0:
            rounded = 100.0
        if values and rounded < values[-1]:
            rounded = values[-1]
        values.append(rounded)
    return values


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []

    at_time = parse_time(case.get("at"))
    target_time = parse_time(case.get("target"))
    if target_time is not None and at_time is not None:
        horizon_hours = (target_time - at_time).total_seconds() / 3600.0
    elif case.get("horizon_minutes") is not None:
        horizon_hours = float(case["horizon_minutes"]) / 60.0
    else:
        horizon_hours = 0.0
    if horizon_hours < 0.0:
        horizon_hours = 0.0
    horizon_days = horizon_hours / 24.0
    floor_sigma = 0.5 * math.sqrt(horizon_days) if horizon_days > 0 else 0.0

    # Time base: hours since the first read (or the checkpoint when the
    # history is empty).
    read_times = []
    for read in history:
        read_times.append(parse_time(read.get("at")))
    if read_times and read_times[0] is not None:
        origin = read_times[0]
    else:
        origin = at_time
    read_hours = []
    for moment in read_times:
        if moment is not None and origin is not None:
            read_hours.append((moment - origin).total_seconds() / 3600.0)
        else:
            read_hours.append(0.0)
    if target_time is not None and origin is not None:
        hours_target = (target_time - origin).total_seconds() / 3600.0
    elif read_hours:
        hours_target = read_hours[-1] + horizon_hours
    else:
        hours_target = horizon_hours

    # Median of the last available rate of every trophy, for fallbacks.
    last_rates = []
    count = len(trophies)
    for idx in range(count):
        found = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                found = float(rates[idx])
                break
        if found is not None:
            last_rates.append(found)
    if last_rates:
        ordered = sorted(last_rates)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            case_median_rate = ordered[mid]
        else:
            case_median_rate = 0.5 * (ordered[mid - 1] + ordered[mid])
    else:
        case_median_rate = None

    rows = []
    for idx, trophy in enumerate(trophies):
        trophy_id = trophy.get("id", idx + 1)

        hours = []
        logits = []
        rates_seen = []
        players_seen = []
        for pos, read in enumerate(history):
            rates = read.get("rates") or []
            if idx >= len(rates):
                continue
            raw = rates[idx]
            if raw is None:
                continue
            rate = float(raw)
            hours.append(read_hours[pos])
            logits.append(logit_of_percent(rate))
            rates_seen.append(rate)
            players_seen.append(read.get("players_tracked"))

        if not history:
            # Empty history: a low, flat, honestly wide forecast.
            median_logit = logit_of_percent(1.0)
            sigma = max(floor_sigma, 1.0)
            if sigma > 2.5:
                sigma = 2.5
            rows.append(
                {"id": trophy_id, "q": quantiles_from(median_logit, sigma)}
            )
            continue

        if not logits:
            # No valid read for this trophy: borrow the case level.
            if case_median_rate is not None:
                median_logit = logit_of_percent(case_median_rate)
            else:
                median_logit = logit_of_percent(1.0)
            sigma = max(2.0 * floor_sigma, 1.25)
            if sigma > 2.5:
                sigma = 2.5
            rows.append(
                {"id": trophy_id, "q": quantiles_from(median_logit, sigma)}
            )
            continue

        direct = direct_path(hours, logits, hours_target)
        direct_logit, resid_sd, step_vol = direct

        structural_logit = structural_path(
            hours, rates_seen, players_seen, hours_target
        )
        if structural_logit is None:
            median_logit = direct_logit
            disagreement = 0.0
        else:
            median_logit = 0.5 * (direct_logit + structural_logit)
            # When the two views disagree, that disagreement is itself
            # evidence about uncertainty: a trophy whose earners trend
            # and whose rate trend point in different directions should
            # not carry the tighter view's confidence.  Half the gap
            # between the paths is therefore added to the spread below.
            disagreement = 0.5 * abs(direct_logit - structural_logit)

        # Keep the blended median inside the representable logit band
        # for a clamped percent, whatever the projections produced.
        low_logit = logit_of_percent(RATE_LO)
        high_logit = logit_of_percent(RATE_HI)
        if median_logit < low_logit:
            median_logit = low_logit
        elif median_logit > high_logit:
            median_logit = high_logit

        horizon_vol = step_vol * math.sqrt(horizon_hours) if horizon_hours > 0 else 0.0
        sigma = math.sqrt(
            resid_sd * resid_sd
            + horizon_vol * horizon_vol
            + disagreement * disagreement
        )
        if sigma < floor_sigma:
            sigma = floor_sigma
        if sigma > 2.5:
            sigma = 2.5

        rows.append(
            {"id": trophy_id, "q": quantiles_from(median_logit, sigma)}
        )

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        names = sorted(os.listdir(input_dir))
    except FileNotFoundError:
        return 0

    for name in names:
        if not name.endswith(".in"):
            continue
        in_path = os.path.join(input_dir, name)
        if not os.path.isfile(in_path):
            continue
        with open(in_path, "r", encoding="utf-8") as handle:
            case = json.load(handle)
        result = forecast_case(case)
        out_name = name[:-3] + ".out"
        out_path = os.path.join(output_dir, out_name)
        payload = json.dumps(result, separators=(",", ":"))
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())

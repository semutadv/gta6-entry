#!/usr/bin/env python3
"""Forecast a game launch day's Twitch viewers, hours 2..24, as quantiles.

Entry for make.fun contest gta6-twitch-ccu-forecast-01 (program/v1).

Method: intraday shape transfer from past big launches.  Every comparable
whose release day is fully observed contributes, for each hour h = 2..24,
the log-ratio ln(viewers_h / viewers_1) of its own release day.  The
forecast for a case with hour-1 print p is p * exp(q_tau(h)), where
q_tau(h) is a weighted empirical quantile of those log-ratios at level
tau, blended 25% toward a lognormal fit (weighted mean/sd of the same
ratios).  Comparables are weighted by

  * a Gaussian kernel on |ln(hour-1 viewers)| distance to the target's
    hour-1 print (sigma = 1.0): launches that were already big in hour 1
    (rolling midnight unlocks, like GTA VI) have a very different
    intraday shape from launches that only unlock later in the UTC day;
  * a factor 3 when the comparable's unlock pattern, parsed from its
    `unlock` text, matches the target's (parsed from the case `notes`):
    rolling midnight-local vs. a single global unlock time.

The weights, kernel width and blend were chosen by leave-one-out
validation over the comparables themselves (each comparable predicted
from all the others, scored with the contest's pinball loss on the log
scale): loss 0.390 vs. 0.930 for the flat house baseline on the same
protocol and 0.714 for the unweighted pooled transfer; on the rolling
midnight-local subset (GTA VI's own pattern) 0.252 vs. 0.357.

Standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
SIGMA_LOG_H1 = 1.0        # kernel width on ln(hour-1 viewers) distance
CLASS_BONUS = 3.0         # weight factor for a matching unlock pattern
SHRINK = 0.25             # blend of empirical quantiles toward lognormal
BASELINE_SPREAD = 0.15    # house baseline spread, used only as a fallback
MAX_Q = 1_000_000_000.0   # output cap from the contest rules


def parse_ts(text):
    """Parse an ISO 8601 timestamp (with Z) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def unlock_class(text):
    """Classify an unlock description: 'rolling' midnight-local or 'other'.

    Returns None when the text says nothing recognisable, in which case
    no class factor is applied.
    """
    if not text:
        return None
    low = text.lower()
    if "midnight local" in low or "rolling" in low:
        return "rolling"
    if "simultaneous" in low or "global unlock" in low or "single unlock" in low:
        return "other"
    return None


def release_day(comp):
    """Mean viewers per hour (1..24) of a comparable's release day.

    Returns None when the release day is not fully observed with
    positive viewers in every hour (such a comparable carries no usable
    intraday shape; the GTA VI category's own premiere history is one).
    """
    try:
        start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    buckets = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None:
            continue
        try:
            delta = (parse_ts(point["from"]) - start).total_seconds()
        except (KeyError, TypeError, ValueError):
            continue
        hour = int(delta // 3600) + 1
        if 1 <= hour <= 24:
            buckets.setdefault(hour, []).append(float(viewers))
    if len(buckets) != 24:
        return None
    day = {h: sum(v) / len(v) for h, v in buckets.items()}
    if any(v <= 0 for v in day.values()):
        return None
    return day


def norm_ppf(p):
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 0.97575:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def weighted_quantile(items, tau):
    """Weighted quantile of (value, weight) items, midpoint-CDF interpolation."""
    items = sorted(items)
    total = sum(w for _, w in items)
    if total <= 0:
        items = [(v, 1.0) for v, _ in items]
        total = float(len(items))
    points = []
    cum = 0.0
    for value, weight in items:
        cum += weight
        points.append(((cum - 0.5 * weight) / total, value))
    if tau <= points[0][0]:
        return points[0][1]
    if tau >= points[-1][0]:
        return points[-1][1]
    for i in range(1, len(points)):
        if tau <= points[i][0]:
            p0, v0 = points[i - 1]
            p1, v1 = points[i]
            if p1 == p0:
                return v1
            return v0 + (v1 - v0) * (tau - p0) / (p1 - p0)
    return points[-1][1]


def baseline_row(p, hour, levels):
    """The house baseline's row for one hour (fallback if no comparables)."""
    spread = BASELINE_SPREAD * math.sqrt(hour - 1)
    return [p * math.exp(norm_ppf(tau) * spread) for tau in levels]


def forecast_case(case):
    """Build the {"hours": [...]} forecast object for one case."""
    levels = case.get("levels") or DEFAULT_LEVELS
    known = case.get("known") or []
    p = 1.0
    if known:
        v = known[0].get("viewers")
        if v is not None and float(v) >= 1.0:
            p = float(v)
    target_class = unlock_class(case.get("notes"))
    log_p = math.log(p)

    # Collect usable comparables: (hour-1 viewers, unlock class, {h: log-ratio}).
    pool = []
    for comp in case.get("comparables") or []:
        day = release_day(comp)
        if day is None:
            continue
        h1 = day[1]
        ratios = {h: math.log(day[h] / h1) for h in range(2, 25)}
        pool.append((h1, unlock_class(comp.get("unlock")), ratios))

    def weight(entry):
        h1, cls, _ = entry
        dist = math.log(h1) - log_p
        w = math.exp(-(dist * dist) / (2.0 * SIGMA_LOG_H1 * SIGMA_LOG_H1))
        if target_class is not None and cls is not None and cls == target_class:
            w *= CLASS_BONUS
        return w

    weights = [weight(entry) for entry in pool]

    rows = []
    forecast = sorted(case.get("forecast") or [], key=lambda r: r["hour"])
    hours = [row["hour"] for row in forecast] or list(range(2, 25))
    for hour in hours:
        if not pool:
            qs = baseline_row(p, hour, levels)
        else:
            items = [(entry[2][hour], w) for entry, w in zip(pool, weights)]
            total_w = sum(weights)
            mean = sum(v * w for v, w in items) / total_w
            var = sum(w * (v - mean) ** 2 for v, w in items) / total_w
            sd = math.sqrt(var)
            qs = []
            for tau in levels:
                emp = weighted_quantile(items, tau)
                logn = mean + norm_ppf(tau) * sd
                qs.append(p * math.exp((1.0 - SHRINK) * emp + SHRINK * logn))
        qs = sorted(min(max(q, 1e-9), MAX_Q) for q in qs)
        rows.append({"hour": hour, "q": [round(q, 4) for q in qs]})
    return {"hours": rows}


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_name = name[:-len(".in")] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

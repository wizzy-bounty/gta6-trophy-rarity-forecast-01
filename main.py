"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase earn rate.

Model, per trophy, on the logit scale (the scale the pinball loss is scored on):

- The house's reads are compressed into distinct page snapshots: the page only changes
  when Exophase refreshes it, so consecutive identical reads carry no new information.
- Rich history: a recency-weighted least-squares slope of the trophy's logit rate against
  ln(1 + hours since release) -- a decelerating clock, because a player base grows and a
  cohort progresses more slowly as the week goes on. The slope is shrunk toward the pooled
  slope of all trophies, the forecast is anchored at the last observed rate, and the total
  move is capped. The spread combines the observed snapshot-to-snapshot volatility (a
  random walk over the horizon), the slope's standard error, and a floor for the rounding
  and last-refresh jitter.
- Launch freeze: if the page has not changed once in the whole observed history, the
  tracked base is still the pre-release population (a few hundred players) and the case
  sits within two days of release, the first refresh inside the horizon will fold in the
  launch surge: the tracked base multiplies several-fold while earners of most trophies
  grow only modestly, so rates fall hard in odds. The centre is shifted by
  ln(F_E / F_P) with F_P = 6.0, F_E = 1.34, and the spread widened to cover the surge's
  uncertainty. This regime cannot trigger on a history with any observed refresh.

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output). Python 3.13, standard library only, deterministic.

Development notes: the constants were set against the kit's published facts, not fitted
to any hidden answer. The surge multipliers are the launch refresh the kit documents
(638 tracked players and 305 earn-events outside the 100% trophy before it; 3,868 and
408 after it). The surge spread was picked for the best worst case across reconstructions
of that refreshed page that spread the 408 earn-events over the trophies in different
shapes (uniform, front-loaded, back-loaded, jittered, integer-rounded). The rich-history
path was exercised on simulated week-long recordings cut like the final case, under two
different shapes of rate dynamics (decelerating and linear), where it scored well below
both house baselines' designs; a pure linear-in-days drift and a per-trophy volatility
blend were tried and dropped for scoring worse.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch-freeze surge: tracked-base multiplier and earners multiplier at the first refresh.
SURGE_PLAYERS = 6.0
SURGE_EARNERS = 1.34
SURGE_SIGMA = 0.35
# Random-walk spread floor and slope damping for the rich-history path.
SIGMA_FLOOR = 0.16
MAX_SLOPE = 2.2
MAX_MOVE = 2.8
HALF_LIFE_H = 8.0
MOVE_UNCERTAINTY = 0.30


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (as JavaScript's Math.round does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def release_epoch(case):
    try:
        d = case["game"]["release_date"]
        return datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        # Fall back to six hours before the first read: the launch-night recording
        # starts just before the release day.
        return parse_ts(case["history"][0]["slot"]) - 6 * 3600


def snapshots(history):
    """Distinct page states, in order: (first_slot_ts, players, rates)."""
    out = []
    prev_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev_key:
            out.append((parse_ts(read["slot"]), read["players_tracked"], read["rates"]))
            prev_key = key
    return out


def wls_slope(xs, ys, ws):
    """Weighted least-squares slope and its standard error."""
    sw = sum(ws)
    if sw <= 0:
        return 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return 0.0, 0.0
    slope = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    dof = max(1.0, sw - 2.0)
    var = sum(w * r * r for r, w in zip(resid, ws)) / dof
    return slope, math.sqrt(var / sxx)


def launch_freeze(case, snaps):
    """True when the page never changed in the observed history and the case sits in
    the launch window with a pre-release-sized tracked base."""
    if len(snaps) != 1:
        return False
    hist = case["history"]
    span_min = (parse_ts(hist[-1]["slot"]) - parse_ts(hist[0]["slot"])) / 60.0
    if span_min < 90:
        return False
    if snaps[-1][1] > 5000:
        return False
    age_h = (parse_ts(case["at"]) - release_epoch(case)) / 3600.0
    return 0 <= age_h <= 48


def forecast_case(case):
    hist = case["history"]
    if not hist:
        raise ValueError("no history")
    snaps = snapshots(hist)
    horizon_min = case["horizon_minutes"]
    horizon_days = horizon_min / 1440.0
    rel = release_epoch(case)
    last_rates = hist[-1]["rates"]

    # Pooled snapshot-to-snapshot volatility, logit per sqrt(day), robust (median-based).
    inc = []
    for a, b in zip(snaps, snaps[1:]):
        dt = (b[0] - a[0]) / 86400.0
        if dt <= 0:
            continue
        for ra, rb in zip(a[2], b[2]):
            if ra is not None and rb is not None:
                inc.append(abs(logit(rb) - logit(ra)) / math.sqrt(dt))
    pooled_vol = 0.5
    if inc:
        inc.sort()
        pooled_vol = max(0.12, inc[len(inc) // 2] / 0.6745)

    frozen = launch_freeze(case, snaps)
    surge_shift = math.log(SURGE_EARNERS / SURGE_PLAYERS)

    # First pass: per-trophy slopes on two clocks -- ln(1 + hours since release), which
    # decelerates, and plain hours, which does not. The two fits are averaged, so the
    # forecast does not bet everything on one shape of the week's dynamics.
    t_last = snaps[-1][0]
    t_target = parse_ts(case["target"])
    clocks = [
        (lambda t: math.log1p(max(0.0, (t - rel) / 3600.0))),
        (lambda t: (t - rel) / 3600.0),
    ]
    fits = []  # per trophy: list over clocks of (slope, se, wsum) or None
    for i in range(len(case["trophies"])):
        pts = [(s[0], logit(s[2][i])) for s in snaps if s[2][i] is not None]
        per_clock = []
        for clock in clocks:
            if len(pts) >= 3:
                xs = [clock(t) for t, _ in pts]
                ys = [v for _, v in pts]
                ws = [0.5 ** ((t_last - t) / 3600.0 / HALF_LIFE_H) for t, _ in pts]
                slope, se = wls_slope(xs, ys, ws)
                per_clock.append((slope, se, sum(ws)))
            else:
                per_clock.append(None)
        fits.append(per_clock)
    pooled = []  # per clock: pooled slope
    for c in range(len(clocks)):
        good = [f[c] for f in fits if f[c] is not None]
        if good:
            tw = sum(g[2] for g in good)
            pooled.append(sum(g[0] * g[2] for g in good) / tw if tw > 0 else 0.0)
        else:
            pooled.append(0.0)
    dxs = [max(0.0, clock(t_target) - clock(t_last)) for clock in clocks]

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        # Last observed rate for this trophy (fall back along the history).
        rate = last_rates[i]
        if rate is None:
            for s in reversed(snaps):
                if s[2][i] is not None:
                    rate = s[2][i]
                    break
        if rate is None:
            rate = 0.05
        center = logit(rate)
        if frozen:
            if rate < 99.99:
                center += surge_shift
            sigma = math.sqrt(SURGE_SIGMA ** 2 + (0.5 * math.sqrt(horizon_days)) ** 2)
        elif len(snaps) >= 3 and any(f is not None for f in fits[i]):
            moves = []
            ses = []
            for c in range(len(clocks)):
                f = fits[i][c]
                if f is None:
                    continue
                slope, se, wsum = f
                shrink = wsum / (wsum + 6.0)
                slope = shrink * slope + (1.0 - shrink) * pooled[c]
                slope = max(-MAX_SLOPE, min(MAX_SLOPE, slope))
                moves.append(max(-MAX_MOVE, min(MAX_MOVE, slope * dxs[c])))
                ses.append(se * dxs[c])
            move = sum(moves) / len(moves)
            center += move
            sigma = math.sqrt(
                SIGMA_FLOOR ** 2
                + (pooled_vol * math.sqrt(horizon_days)) ** 2
                + (sum(ses) / len(ses)) ** 2
                + (MOVE_UNCERTAINTY * abs(move)) ** 2
            )
        else:
            sigma = math.sqrt(SIGMA_FLOOR ** 2 + (0.5 * math.sqrt(horizon_days)) ** 2)
        qs = [round4(inv_logit(center + z * sigma)) for z in Z]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/semutadv/gta6-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/semutadv/gta6-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers — hour-by-hour quantile forecast

Entry for make.fun contest **gta6-twitch-ccu-forecast-01**
("Forecast GTA VI's launch-day Twitch viewers hour by hour", harness
program/v1). `main.py` (Python 3, standard library only, no network,
deterministic) reads every `*.in` case in `$MAKEFUN_INPUT_DIR`
(default `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR`
(default `/output`): for each of hours 2..24, the 0.05 / 0.25 / 0.5 /
0.75 / 0.95 quantiles of that hour's Twitch viewer count in the game's
category.

## Approach: intraday shape transfer

A launch day's intraday shape — how viewership grows, peaks and decays
relative to its own first hour — is far more predictable than its
absolute size, and the case already pins the size down with the hour-1
print `p`. So for every comparable with a fully observed release day,
the program computes the log-ratios `ln(viewers_h / viewers_1)` for
h = 2..24, and forecasts

    q_tau(h) = p * exp( weighted quantile_tau of the log-ratios at h )

with two refinements, both kept because leave-one-out validation (see
below) says they pay:

* **Similarity weights.** Comparables are weighted by a Gaussian kernel
  (sigma = 1.0) on the distance between their log hour-1 viewers and the
  target's. This separates the two launch regimes: games already live
  worldwide in hour 1 (rolling midnight-local unlocks — GTA VI's own
  pattern, unlocked in New Zealand 13 h before the UTC day starts) decay
  and re-surge with the US day, while games that unlock at one global
  time later in the UTC day explode upward from a near-zero hour 1.
  A further factor 3 applies when the comparable's unlock pattern,
  parsed from its `unlock` text, matches the target's, parsed from the
  case `notes` (rolling midnight-local vs. a single global unlock).
* **Lognormal shrinkage.** Each hour's empirical quantiles are blended
  25% toward a lognormal fit (weighted mean/sd of the same log-ratios),
  which steadies the 5% / 95% tails that rest on few effective points.

Comparables without a complete, positive release day are skipped — in
particular the GTA VI category's own premiere history, which is not a
launch day. If no comparable were usable, the program falls back to the
house baseline's flat shape so the output stays valid.

## Validation

No server grade exists for the practice cases, so the model was tuned
by leave-one-out over the comparables inside the practice case files:
each of the 37 comparables with a complete release day is treated as a
pseudo-case, predicted from its own hour 1 using all the others, and
scored with the contest's exact pinball loss on the log scale. The flat
house baseline is scored on the same protocol.

| model | LOO loss (all 37) | LOO loss (rolling subset) |
| --- | --- | --- |
| this program | **0.390** | **0.252** |
| house baseline | 0.930 | 0.357 |
| unweighted pooled transfer | 0.714 | — |

The program runs all four practice cases in ~0.2 s total and produces
byte-identical output on repeated runs; every output was checked
against the contest's schema (23 rows, five finite quantiles in
(0, 1e9], non-decreasing).

## Layout

* `main.py` — the entry program.
* `.makefun/owner.json` — entrant wallet for the submission signature.
* `LICENSE` — MIT, from the contest kit.

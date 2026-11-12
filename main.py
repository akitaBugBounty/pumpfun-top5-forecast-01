# pumpfun-top5-forecast-01: Monte Carlo top-5 forecaster (Python 3.13, stdlib only).
# Reads /input/task.json + /input/history.ndjson, writes /output/forecast.json.
# Method: per-coin log10-cap drift + volatility + collapse hazard, damped with age;
# seeded Monte Carlo over the 7 answer reads in task.json; newcomer mass left unnamed.
# v2 (tuned on local holdouts vs the house persistence baseline): drift x0.3 and
# vol x0.5 — 1-5 obs per coin make raw drift/vol far too noisy to trust at face
# value — and a larger unnamed reserve (base 0.25, cap 0.55) for movers/newcomers.
import json, math, os, random

IN = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUT = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

def load_task():
    with open(os.path.join(IN, "task.json")) as f:
        return json.load(f)

def load_marks():
    marks = []
    with open(os.path.join(IN, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") and isinstance(m.get("rows"), list):
                marks.append(m)
    return marks

def cap_sol(row):
    return float(row["market_cap_lamports"]) / 1e9

def logcap(row):
    return math.log10(cap_sol(row))

def _parse_time(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

def build_coins(marks):
    traj = {}
    for m in marks:
        t = _parse_time(m["at"])
        for r in m["rows"]:
            c = traj.setdefault(r["mint"], {"logs": [], "rank": 99, "venue": "curve"})
            c["logs"].append(logcap(r))
            c["rank"] = r["rank"]
            c["venue"] = r.get("venue", "curve")
    return traj

def global_stats(marks, traj):
    vanish = []
    for a, b in zip(marks, marks[1:]):
        top = set(r["mint"] for r in a["rows"][:50])
        nxt = set(r["mint"] for r in b["rows"])
        if top:
            vanish.append(len(top - nxt) / len(top))
    v = sum(vanish) / len(vanish) if vanish else 0.15
    rets = []
    for c in traj.values():
        ls = c["logs"]
        rets += [y - x for x, y in zip(ls, ls[1:])]
    mu = sum(rets) / len(rets) if rets else 0.0
    sd = (sum((x - mu) ** 2 for x in rets) / len(rets)) ** 0.5 if len(rets) > 1 else 0.12
    return max(0.02, min(0.5, v)), max(0.05, min(0.30, sd))

def coin_params(c, vol_step, vanish):
    ls = c["logs"]
    n = len(ls) - 1
    if n >= 1:
        rs = [y - x for x, y in zip(ls, ls[1:])]
        mu = sum(rs) / n
        sd = (sum((x - mu) ** 2 for x in rs) / n) ** 0.5 if n > 1 else vol_step
    else:
        mu, sd = 0.0, vol_step
    drift = max(-0.6, min(0.6, mu * 6 * n / (n + 3))) * 0.3
    vol = max(0.15, min(0.70, sd * 6 ** 0.5)) * 0.5
    lam = min(0.35, 0.10 + 0.5 * vanish)
    if drift > 0.1:
        lam *= 0.5
    elif drift < -0.1:
        lam *= 1.5
    if c["rank"] <= 5:
        lam *= 0.6
    return {"x0": ls[-1], "drift": drift, "vol": vol, "lam": lam, "rank": c["rank"]}

N_PATHS = 2500
SEED = 20261112

def simulate(params, coins, read_hours):
    rng = random.Random(SEED)
    n = len(coins)
    read_steps = [int(round(h * 6)) for h in read_hours]
    steps = max(read_steps) if read_steps else 0
    place_counts = {k: {} for k in range(1, 6)}
    size_samp = {k: [] for k in range(1, 6)}
    for _ in range(N_PATHS):
        xs = [p["x0"] for p in params]
        dead = [False] * n
        step = 0
        for target in read_steps:
            while step < target:
                step += 1
                damp = math.exp(-(step / 6.0) / 8.0)
                for i in range(n):
                    if dead[i]:
                        continue
                    p = params[i]
                    if rng.random() < 1 - math.exp(-p["lam"] / 6.0):
                        xs[i] = rng.gauss(1.7, 0.3)
                        dead[i] = True
                    else:
                        xs[i] += p["drift"] * damp / 6.0 + rng.gauss(0, p["vol"] / 6 ** 0.5)
            order = sorted(range(n), key=lambda i: -xs[i])
            for k in range(1, 6):
                m = coins[order[k - 1]]
                place_counts[k][m] = place_counts[k].get(m, 0) + 1
                size_samp[k].append(xs[order[k - 1]])
    return place_counts, size_samp

def _erfc(x):
    z = abs(x)
    t = 1 / (1 + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2 - r

def _above(x, mean, sd):
    if x == -float("inf"):
        return 1.0
    if x == float("inf"):
        return 0.0
    return 0.5 * _erfc((x - mean) / (sd * math.sqrt(2)))

def _floor6(p):
    return math.floor(p * 1e6) / 1e6

def assemble(task, place_counts, size_samp, read_hours):
    edges = task["size"]["edges"]
    mean_h = sum(read_hours) / len(read_hours) if read_hours else 0.0
    base = min(0.55, 0.25 + 0.035 * mean_h)
    probs = []
    sizes = []
    for k in range(1, 6):
        reserve = min(0.55, base + 0.04 * (k - 1))
        counts = place_counts[k]
        total = sum(counts.values())
        cand = sorted(((c / total, m) for m, c in counts.items()), reverse=True)
        budget = 1.0 - reserve
        used = 0.0
        named = 0
        for p, m in cand:
            alloc = _floor6(min(p, budget - used))
            share = reserve / max(1, 100 - named)
            if alloc > share and alloc > 0:
                probs.append({"mint": m, "place": k, "p": alloc})
                used += alloc
                named += 1
        xs = size_samp[k]
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        sd = max(0.25, var ** 0.5)
        bins = []
        for b in range(len(edges) + 1):
            lo = -float("inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == len(edges) else edges[b]
            bins.append((_floor6(_above(lo, mean, sd) - _above(hi, mean, sd)), b))
        bins.sort(reverse=True)
        sused = 0.0
        snamed = 0
        for p, b in bins:
            alloc = _floor6(min(p, 0.85 - sused))
            share = 0.15 / max(1, 82 - snamed)
            if alloc > 0 and alloc > share:
                sizes.append({"place": k, "bin": b, "p": alloc})
                sused += alloc
                snamed += 1
    score = {}
    for k in range(1, 6):
        total = sum(place_counts[k].values())
        for m, c in place_counts[k].items():
            score[m] = score.get(m, 0.0) + (c / total) * (6 - k)
    top5 = [m for m, _ in sorted(score.items(), key=lambda kv: -kv[1])[:5]]
    return probs, sizes, top5

def main():
    task = load_task()
    marks = load_marks()
    if not marks:
        raise SystemExit("no marks")
    traj = build_coins(marks)
    vanish, vol_step = global_stats(marks, traj)
    last = marks[-1]
    coins = [r["mint"] for r in sorted(last["rows"], key=lambda r: r["rank"])[:180]]
    params = [coin_params(traj[m], vol_step, vanish) for m in coins]
    t_last = _parse_time(last["at"])
    read_hours = [(_parse_time(s) - t_last) / 3600.0 for s in task["reads"]]
    place_counts, size_samp = simulate(params, coins, read_hours)
    probs, sizes, top5 = assemble(task, place_counts, size_samp, read_hours)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, task["output"]), "w") as f:
        f.write(json.dumps({"probabilities": probs, "top5": top5, "sizes": sizes}) + "\n")

if __name__ == "__main__":
    main()

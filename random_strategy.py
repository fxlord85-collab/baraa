"""مولّد استراتيجية تداول عشوائية + اختبار رجعي على بيانات اصطناعية.

الاستخدام:
    python random_strategy.py [--seed N] [--bars N]
"""
import argparse
import math
import random


def sma(xs, n, i):
    return sum(xs[i - n + 1:i + 1]) / n


def rsi(xs, n, i):
    gains = losses = 0.0
    for k in range(i - n + 1, i + 1):
        d = xs[k] - xs[k - 1]
        gains += max(d, 0)
        losses += max(-d, 0)
    return 100.0 if losses == 0 else 100 - 100 / (1 + gains / losses)


def random_strategy(rng):
    """يختار نوع المؤشر ومعاملاته ووقف الخسارة/جني الربح عشوائياً."""
    kind = rng.choice(["sma_cross", "rsi_reversion", "breakout"])
    params = {"kind": kind,
              "stop_loss": round(rng.uniform(0.01, 0.05), 3),
              "take_profit": round(rng.uniform(0.02, 0.10), 3)}
    if kind == "sma_cross":
        fast = rng.randint(5, 20)
        params.update(fast=fast, slow=rng.randint(fast + 5, 60))
    elif kind == "rsi_reversion":
        params.update(period=rng.randint(7, 21),
                      buy_below=rng.randint(20, 35),
                      sell_above=rng.randint(65, 80))
    else:
        params.update(lookback=rng.randint(10, 50))
    return params


def warmup(p):
    return {"sma_cross": p.get("slow", 0),
            "rsi_reversion": p.get("period", 0) + 1,
            "breakout": p.get("lookback", 0) + 1}[p["kind"]]


def signal(p, px, i):
    """1 = شراء، -1 = خروج، 0 = لا شيء."""
    if p["kind"] == "sma_cross":
        f, s = sma(px, p["fast"], i), sma(px, p["slow"], i)
        pf, ps = sma(px, p["fast"], i - 1), sma(px, p["slow"], i - 1)
        return 1 if pf <= ps and f > s else -1 if pf >= ps and f < s else 0
    if p["kind"] == "rsi_reversion":
        r = rsi(px, p["period"], i)
        return 1 if r < p["buy_below"] else -1 if r > p["sell_above"] else 0
    window = px[i - p["lookback"]:i]
    return 1 if px[i] > max(window) else -1 if px[i] < min(window) else 0


def gen_prices(rng, bars, drift=0.0003, vol=0.015, start=100.0):
    px = [start]
    for _ in range(bars - 1):
        px.append(px[-1] * math.exp(rng.gauss(drift, vol)))
    return px


def backtest(p, px, fee=0.001):
    equity, peak, max_dd = 1.0, 1.0, 0.0
    entry = None
    trades, wins = 0, 0
    for i in range(warmup(p), len(px)):
        if entry is not None:
            ret = px[i] / entry - 1
            if ret <= -p["stop_loss"] or ret >= p["take_profit"] or signal(p, px, i) == -1:
                equity *= (1 + ret) * (1 - fee) ** 2
                trades += 1
                wins += ret > 0
                entry = None
        elif signal(p, px, i) == 1:
            entry = px[i]
        peak = max(peak, equity)
        max_dd = max(max_dd, 1 - equity / peak)
    return {"return": equity - 1, "trades": trades,
            "win_rate": wins / trades if trades else 0.0, "max_drawdown": max_dd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bars", type=int, default=1000)
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else random.randrange(10**6)
    rng = random.Random(seed)
    p = random_strategy(rng)
    px = gen_prices(rng, a.bars)
    r = backtest(p, px)
    print(f"seed={seed}")
    print("الاستراتيجية:", p)
    print(f"العائد: {r['return']:.2%} | الصفقات: {r['trades']} | "
          f"نسبة الربح: {r['win_rate']:.0%} | أقصى تراجع: {r['max_drawdown']:.2%}")
    print(f"(buy&hold: {px[-1] / px[warmup(p)] - 1:.2%})")


if __name__ == "__main__":
    main()

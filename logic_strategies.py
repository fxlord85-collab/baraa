"""استراتيجيات مبنية على منطق، تُختبر على شموع GBPUSD المحفوظة في الكاش.

1) trend_pullback: تداول مع الاتجاه (SMA طويل) عند تراجع مؤقت (RSI)، بوقف متحرك حسب التذبذب.
2) session_breakout: اختراق نطاق الجلسة الآسيوية خلال جلسة لندن، بفلتر اتجاه اختياري.

python logic_strategies.py --cache bars_1h.parquet
(الاستراتيجية الثانية تفترض أن توقيت الشموع UTC وأن الإطار 1h)
"""
import argparse
import itertools


class Ctx:
    """يحسب المؤشرات مرة واحدة لكل معامل."""

    def __init__(self, px):
        self.px, self._m = px, {}

    def _get(self, key, fn, n):
        if (key, n) not in self._m:
            self._m[(key, n)] = fn(self.px, n)
        return self._m[(key, n)]

    def sma(self, n):
        return self._get("sma", sma_series, n)

    def rsi(self, n):
        return self._get("rsi", rsi_series, n)

    def atr(self, n):
        return self._get("atr", atr_series, n)


def sma_series(xs, n):
    out, c = [None] * len(xs), 0.0
    for i, x in enumerate(xs):
        c += x
        if i >= n:
            c -= xs[i - n]
        if i >= n - 1:
            out[i] = c / n
    return out


def rsi_series(xs, n):
    out, ag, al = [None] * len(xs), 0.0, 0.0
    for i in range(1, len(xs)):
        d = xs[i] - xs[i - 1]
        g, l = max(d, 0.0), max(-d, 0.0)
        if i <= n:
            ag, al = ag + g / n, al + l / n
        else:
            ag, al = (ag * (n - 1) + g) / n, (al * (n - 1) + l) / n
        if i >= n:
            out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def atr_series(xs, n):
    """متوسط التغير المطلق (بديل ATR لأن لدينا الإغلاق فقط)."""
    out, c = [None] * len(xs), 0.0
    for i in range(1, len(xs)):
        c += abs(xs[i] - xs[i - 1])
        if i > n:
            c -= abs(xs[i - n] - xs[i - n - 1])
        if i >= n:
            out[i] = c / n
    return out


def trend_pullback(px, ctx, p, fee, lo, hi, **_):
    sm, r, a = ctx.sma(p["trend_n"]), ctx.rsi(p["rsi_n"]), ctx.atr(p["atr_n"])
    first = max(p["trend_n"], p["rsi_n"], p["atr_n"]) + 1
    trades, pos, entry, best = [], 0, 0.0, 0.0
    for i in range(max(lo, first), hi):
        x = px[i]
        if pos:
            if pos > 0:
                best = max(best, x)
                hit = x <= best - p["mult"] * a[i]
                back = r[i] >= 50
            else:
                best = min(best, x)
                hit = x >= best + p["mult"] * a[i]
                back = r[i] <= 50
            if hit or back:
                trades.append((i, pos * (x - entry) / entry - 2 * fee))
                pos = 0
        elif x > sm[i] and r[i] < p["entry_lo"]:
            pos, entry, best = 1, x, x
        elif x < sm[i] and r[i] > 100 - p["entry_lo"]:
            pos, entry, best = -1, x, x
    if pos:
        trades.append((hi - 1, pos * (px[hi - 1] - entry) / entry - 2 * fee))
    return trades


def session_breakout(px, ctx, p, fee, lo, hi, hours, days, **_):
    sm = ctx.sma(p["trend_n"]) if p["use_trend"] else None
    first = (p["trend_n"] + 1) if p["use_trend"] else 1
    trades, pos, entry, stop = [], 0, 0.0, 0.0
    day, ahi, alo, traded = None, 0.0, 0.0, False
    for i in range(max(lo, first), hi):
        h, d, x = hours[i], days[i], px[i]
        if d != day:
            if pos:
                trades.append((i - 1, pos * (px[i - 1] - entry) / entry - 2 * fee))
                pos = 0
            day, ahi, alo, traded = d, float("-inf"), float("inf"), False
        if h < p["asia_end"]:
            ahi, alo = max(ahi, x), min(alo, x)
            continue
        if pos:
            if h >= p["exit_hour"] or (pos > 0 and x <= stop) or (pos < 0 and x >= stop):
                trades.append((i, pos * (x - entry) / entry - 2 * fee))
                pos = 0
            continue
        if traded or h >= p["exit_hour"] or ahi == float("-inf"):
            continue
        mid = (ahi + alo) / 2
        if x > ahi and (sm is None or x > sm[i]):
            pos, entry, traded = 1, x, True
            stop = mid if p["stop"] == "mid" else alo
        elif x < alo and (sm is None or x < sm[i]):
            pos, entry, traded = -1, x, True
            stop = mid if p["stop"] == "mid" else ahi
    if pos:
        trades.append((hi - 1, pos * (px[hi - 1] - entry) / entry - 2 * fee))
    return trades


def metrics(trades, years):
    eq = peak = 1.0
    dd, wins, by_year = 0.0, 0, {}
    for i, r in trades:
        eq *= 1 + r
        peak = max(peak, eq)
        dd = max(dd, 1 - eq / peak)
        wins += r > 0
        y = years[i]
        by_year[y] = by_year.get(y, 1.0) * (1 + r)
    n = len(trades)
    return {"ret": eq - 1, "trades": n, "win": wins / n if n else 0.0, "dd": dd,
            "years": {y: v - 1 for y, v in sorted(by_year.items())}}


def grid(space):
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*space.values())]


STRATEGIES = {
    "trend_pullback": (trend_pullback, grid({
        "trend_n": [100, 200, 400], "rsi_n": [7, 14], "entry_lo": [30, 35, 40],
        "atr_n": [14, 24], "mult": [2, 3, 4]})),
    "session_breakout": (session_breakout, grid({
        "asia_end": [6, 7, 8], "exit_hour": [14, 16, 18], "use_trend": [True, False],
        "trend_n": [200], "stop": ["mid", "opp"]})),
}


def run(px, hours, days, years, fee, train=0.7, min_trades=30):
    cut = int(len(px) * train)
    ctx = Ctx(px)
    kw = dict(hours=hours, days=days)
    for name, (fn, combos) in STRATEGIES.items():
        rows = []
        for p in combos:
            tr = metrics(fn(px, ctx, p, fee, 0, cut, **kw), years)
            te = metrics(fn(px, ctx, p, fee, cut, len(px), **kw), years)
            rows.append((tr, te, p))
        ok = [r for r in rows if r[0]["trades"] >= min_trades] or rows
        ok.sort(key=lambda r: r[0]["ret"], reverse=True)
        print(f"\n=== {name} ({len(combos)} تركيبة) ===")
        for tr, te, p in ok[:3]:
            print(f"train {tr['ret']:+.1%} ({tr['trades']} صفقة) | "
                  f"test {te['ret']:+.1%} ({te['trades']} صفقة، ربح {te['win']:.0%}، "
                  f"تراجع {te['dd']:.1%}) | {p}")
        best = ok[0][1]
        print("أفضل تركيبة، العائد السنوي خارج العينة:",
              {y: f"{v:+.1%}" for y, v in best["years"].items()})
        tests = [r[1]["ret"] for r in rows]
        print(f"كل التركيبات خارج العينة: متوسط {sum(tests) / len(tests):+.2%} | "
              f"رابحة {sum(t > 0 for t in tests) / len(tests):.0%}")
    print(f"\nشراء والاحتفاظ في فترة الاختبار: {px[-1] / px[cut] - 1:+.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="bars_1h.parquet")
    ap.add_argument("--train", type=float, default=0.7)
    a = ap.parse_args()
    import pandas as pd
    d = pd.read_parquet(a.cache)
    idx = pd.DatetimeIndex(d.index)
    px = d["px"].tolist()
    fee = float(d["spread"].iloc[0]) / 2
    print(f"الشموع: {len(px)} | {idx[0]} -> {idx[-1]} | سبريد {fee * 2e4:.2f} نقطة أساس")
    run(px, idx.hour.tolist(), (idx.year * 1000 + idx.dayofyear).tolist(),
        idx.year.tolist(), fee, a.train)


if __name__ == "__main__":
    main()

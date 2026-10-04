"""اختبار استراتيجيات عشوائية على تكّات parquet حقيقية (GBPUSD).

pip install pandas pyarrow
python backtest_ticks.py --path "F:\\Raw date\\parquet_ticks\\GBPUSD" --tf 5min --n 200
"""
import argparse
import glob
import os
import random

import pandas as pd

from random_strategy import backtest, random_strategy, warmup

TS_NAMES = ("timestamp", "time", "datetime", "date", "ts")
BID_NAMES = ("bid", "bid_price", "b")
ASK_NAMES = ("ask", "ask_price", "a")
PRICE_NAMES = ("price", "last", "close", "mid")


def pick(cols, names):
    low = {c.lower(): c for c in cols}
    return next((low[n] for n in names if n in low), None)


def load_bars(path, tf):
    files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"لا توجد ملفات parquet في: {path}")
    print(f"عدد الملفات: {len(files)}")
    parts, spreads = [], []
    for f in files:
        d = pd.read_parquet(f)
        ts = pick(d.columns, TS_NAMES)
        if ts is None:
            d = d.reset_index()
            ts = pick(d.columns, TS_NAMES) or d.columns[0]
        bid, ask = pick(d.columns, BID_NAMES), pick(d.columns, ASK_NAMES)
        if bid and ask:
            mid = (d[bid] + d[ask]) / 2
            spreads.append(((d[ask] - d[bid]) / mid).mean())
        else:
            p = pick(d.columns, PRICE_NAMES)
            if p is None:
                raise SystemExit(f"أعمدة غير معروفة: {list(d.columns)}")
            mid = d[p]
        t = pd.to_datetime(d[ts], utc=True)
        parts.append(mid.groupby(t.dt.floor(tf)).last())
    bars = pd.concat(parts).groupby(level=0).last().sort_index().dropna()
    spread = sum(spreads) / len(spreads) if spreads else 0.0001
    return bars, spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=r"F:\Raw date\parquet_ticks\GBPUSD")
    ap.add_argument("--tf", default="5min")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--train", type=float, default=0.7)
    a = ap.parse_args()

    bars, spread = load_bars(a.path, a.tf)
    px = bars.tolist()
    cut = int(len(px) * a.train)
    fee = spread / 2  # نصف السبريد لكل جانب
    print(f"الشموع: {len(px)} | {bars.index[0]} -> {bars.index[-1]} | "
          f"السبريد المتوسط: {spread * 1e4:.2f} نقطة أساس")

    rng = random.Random(a.seed)
    rows = []
    for _ in range(a.n):
        p = random_strategy(rng)
        tr = backtest(p, px[:cut], fee)
        te = backtest(p, px[cut - warmup(p):], fee)
        rows.append((tr["return"], te["return"], te["trades"], te["max_drawdown"], p))
    rows.sort(key=lambda r: r[0], reverse=True)

    print("\nأفضل 10 على فترة التدريب، ونتيجتها على فترة الاختبار (خارج العينة):")
    for tr, te, n, dd, p in rows[:10]:
        print(f"train {tr:+.1%} | test {te:+.1%} | صفقات {n} | تراجع {dd:.1%} | {p}")
    pos = sum(r[1] > 0 for r in rows) / len(rows)
    avg = sum(r[1] for r in rows) / len(rows)
    print(f"\nكل الاستراتيجيات خارج العينة: متوسط {avg:+.2%} | نسبة الرابحة {pos:.0%}")
    bh = px[-1] / px[cut] - 1
    print(f"شراء والاحتفاظ في فترة الاختبار: {bh:+.2%}")


if __name__ == "__main__":
    main()

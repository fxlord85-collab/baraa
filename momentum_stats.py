"""فاحص الزخم (time-series momentum) على الإطار اليومي والأسبوعي/الشهري.

لكل تركيبة (فترة النظر L، فترة الاحتفاظ H) بالأيام: إن كان عائد الماضي موجباً نشتري وإلا نبيع،
ونقيس عائد فترة الاحتفاظ التالية. العيّنات غير متداخلة (كل H يوماً) حتى تكون t صحيحة.

python momentum_stats.py --cache bars_1h.parquet

عمودا (خام) و(بدون الانجراف): الثاني يطرح متوسط العائد العام، حتى لا يبدو الانحياز
للشراء/البيع في اتجاه عام للسعر زخماً. صافي التكلفة يخصم السبريد عند كل انقلاب في الاتجاه.
"""
import argparse
import math

LOOKBACKS = (5, 10, 21, 63, 126, 252)  # أيام عمل: أسبوع، أسبوعان، شهر، ربع، نصف سنة، سنة
HOLDS = (5, 21)                         # أسبوع، شهر
T_MIN = 2.9                             # تصحيح لنحو 12 اختباراً


def mean_t(xs):
    n = len(xs)
    if n < 3:
        return 0.0, 0.0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, (m / (sd / math.sqrt(n)) if sd else 0.0)


def momentum_table(px, years, cost_bps, lookbacks=LOOKBACKS, holds=HOLDS):
    rows = []
    for L in lookbacks:
        for H in holds:
            idx = list(range(L, len(px) - H, H))
            if len(idx) < 40:
                continue
            fwd = [1e4 * math.log(px[i + H] / px[i]) for i in idx]
            sign = [1 if px[i] > px[i - L] else -1 for i in idx]
            mf = sum(fwd) / len(fwd)
            raw = [s * f for s, f in zip(sign, fwd)]
            dm = [s * (f - mf) for s, f in zip(sign, fwd)]
            n = len(idx)
            flips = sum(sign[k] != sign[k - 1] for k in range(1, n)) / (n - 1)
            m_raw, t_raw = mean_t(raw)
            m_dm, t_dm = mean_t(dm)
            h1, h2 = mean_t(dm[:n // 2])[0], mean_t(dm[n // 2:])[0]
            by_year = {}
            for i, v in zip(idx, dm):
                by_year.setdefault(years[i], []).append(v)
            ym = [sum(v) / len(v) for v in by_year.values()]
            share = sum(1 for v in ym if v * m_dm > 0) / len(ym)
            net = m_raw - cost_bps * flips
            solid = abs(t_dm) >= T_MIN and h1 * h2 > 0 and share >= 0.7
            verdict = ("قابلة للتداول؟" if solid and net > 0 else
                       "أقل من التكلفة" if solid else "ضجيج")
            rows.append(dict(L=L, H=H, n=n, long=sum(s > 0 for s in sign) / n,
                             raw=m_raw, t_raw=t_raw, dm=m_dm, t_dm=t_dm, h1=h1, h2=h2,
                             share=share, flips=flips, net=net, verdict=verdict))
    return rows


def report(px, years, cost_bps):
    rows = momentum_table(px, years, cost_bps)
    print(f"أيام: {len(px)} | تكلفة الانقلاب ≈ {cost_bps:.2f} نقطة أساس | حد الدلالة |t| ≥ {T_MIN}\n")
    print("نظر | احتفاظ |  n  | %شراء | خام (t)        | بدون الانجراف (t) | نصف1 | نصف2 | سنوات | انقلاب | صافي    | الحكم")
    for r in rows:
        print(f"{r['L']:>4} | {r['H']:>6} | {r['n']:>3} | {r['long']:>4.0%} | "
              f"{r['raw']:+7.1f} ({r['t_raw']:+4.1f}) | {r['dm']:+7.1f} ({r['t_dm']:+4.1f})    | "
              f"{r['h1']:+5.1f} | {r['h2']:+5.1f} | {r['share']:>4.0%} | {r['flips']:>5.0%} | "
              f"{r['net']:+7.1f} | {r['verdict']}")
    good = [r for r in rows if r["verdict"] == "قابلة للتداول؟"]
    print(f"\nتركيبات قابلة للتداول؟ {len(good)} من {len(rows)} (القيم بالنقاط الأساسية لكل فترة احتفاظ)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="bars_1h.parquet")
    a = ap.parse_args()
    import pandas as pd
    d = pd.read_parquet(a.cache)
    s = d["px"].copy()
    s.index = pd.DatetimeIndex(s.index)
    daily = s.resample("1D").last().dropna()
    daily = daily[daily.index.dayofweek < 5]
    print(f"{daily.index[0].date()} -> {daily.index[-1].date()}")
    report(daily.tolist(), daily.index.year.tolist(), float(d["spread"].iloc[0]) * 1e4)


if __name__ == "__main__":
    main()

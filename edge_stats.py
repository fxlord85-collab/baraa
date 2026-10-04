"""فحص خام لوجود أفضلية: متوسط عائد كل ساعة/يوم، والارتباط الذاتي، قبل بناء أي استراتيجية.

python edge_stats.py --cache bars_1h.parquet

كل نتيجة تُقارن بالتكلفة (السبريد ذهاباً وإياباً) وبحدّ دلالة إحصائية مصحّح للاختبارات المتعددة
(|t| >= 3.2)، وتُختبر ثباتها بين نصفَي البيانات وعبر السنوات.
"""
import argparse
import math

T_MIN = 3.2  # تصحيح للاختبارات المتعددة (~31 اختباراً)


def stats(rs):
    n = len(rs)
    if n < 2:
        return 0.0, 0.0, n
    m = sum(rs) / n
    sd = math.sqrt(sum((r - m) ** 2 for r in rs) / (n - 1))
    return m, (m / (sd / math.sqrt(n)) if sd else 0.0), n


def returns(px, hours):
    """عائد كل ساعة بالنقاط الأساسية، يُهمل ما بعد الفجوات (العطلات)."""
    out = [None] * len(px)
    for i in range(1, len(px)):
        if hours[i] == (hours[i - 1] + 1) % 24:
            out[i] = 1e4 * math.log(px[i] / px[i - 1])
    return out


def group(rets, keys, idx):
    g = {}
    for i in idx:
        if rets[i] is not None:
            g.setdefault(keys[i], []).append(rets[i])
    return g


def autocorr(rets, lag, idx):
    xs = [(rets[i], rets[i - lag]) for i in idx
          if i >= lag and rets[i] is not None and rets[i - lag] is not None]
    n = len(xs)
    if n < 10:
        return 0.0, 0.0
    ma, mb = sum(a for a, _ in xs) / n, sum(b for _, b in xs) / n
    cov = sum((a - ma) * (b - mb) for a, b in xs)
    va, vb = sum((a - ma) ** 2 for a, _ in xs), sum((b - mb) ** 2 for _, b in xs)
    r = cov / math.sqrt(va * vb) if va and vb else 0.0
    return r, r * math.sqrt(n)


def report(px, hours, weekdays, years, cost_bps):
    rets = returns(px, hours)
    N = len(px)
    half = N // 2
    halves = (range(0, half), range(half, N))
    print(f"تكلفة الصفقة ذهاباً وإياباً ≈ {cost_bps:.2f} نقطة أساس | حد الدلالة |t| ≥ {T_MIN}\n")

    for title, keys, labels in (("ساعة UTC", hours, range(24)),
                                ("يوم الأسبوع (0=الاثنين)", weekdays, range(5))):
        print(f"--- متوسط العائد حسب {title} (نقطة أساس لكل ساعة) ---")
        print("المفتاح |   كل البيانات (t)  | النصف1 | النصف2 | سنوات بنفس الإشارة | الحكم")
        full = group(rets, keys, range(N))
        h1, h2 = (group(rets, keys, r) for r in halves)
        yrs = {}
        for i in range(N):
            if rets[i] is not None:
                yrs.setdefault((keys[i], years[i]), []).append(rets[i])
        for k in labels:
            if k not in full:
                continue
            m, t, n = stats(full[k])
            m1, m2 = stats(h1.get(k, []))[0], stats(h2.get(k, []))[0]
            ym = [stats(v)[0] for (kk, _), v in yrs.items() if kk == k]
            same = sum(1 for v in ym if v * m > 0)
            share = same / len(ym) if ym else 0
            solid = abs(t) >= T_MIN and m1 * m2 > 0 and share >= 0.7
            tradable = solid and abs(m) > cost_bps
            verdict = ("قابلة للتداول؟" if tradable else
                       "أقل من التكلفة" if solid else "ضجيج")
            print(f"{k:>6} | {m:+7.3f} ({t:+5.1f}) | {m1:+6.2f} | {m2:+6.2f} | "
                  f"{same:>2}/{len(ym):<2} ({share:.0%})        | {verdict}")
        print()

    print("--- الارتباط الذاتي لعوائد الساعة (موجب = استمرار، سالب = ارتداد) ---")
    for lag in (1, 2, 3, 6, 12, 24):
        r, t = autocorr(rets, lag, range(N))
        r1, _ = autocorr(rets, lag, halves[0])
        r2, _ = autocorr(rets, lag, halves[1])
        flag = "دال" if abs(t) >= T_MIN and r1 * r2 > 0 else "ضجيج"
        print(f"lag {lag:>2}: {r:+.4f} (t={t:+5.1f}) | نصف1 {r1:+.4f} | نصف2 {r2:+.4f} | {flag}")
    print("\nملاحظة: حتى الارتباط الدال والثابت قد يكون أصغر من التكلفة؛ قارنه دائماً بها.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="bars_1h.parquet")
    a = ap.parse_args()
    import pandas as pd
    d = pd.read_parquet(a.cache)
    idx = pd.DatetimeIndex(d.index)
    px = d["px"].tolist()
    cost = float(d["spread"].iloc[0]) * 1e4
    print(f"الشموع: {len(px)} | {idx[0]} -> {idx[-1]}")
    report(px, idx.hour.tolist(), idx.dayofweek.tolist(), idx.year.tolist(), cost)


if __name__ == "__main__":
    main()

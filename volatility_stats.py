"""فاحص التقلب: هل يمكن توقعه؟ وماذا يعني ذلك لتحجيم المراكز وتوقيت الساعات.

python volatility_stats.py --cache bars_1h.parquet

1) استمرارية التقلب (الارتباط الذاتي لـ log التقلب اليومي).
2) توقع تقلب اليوم التالي: تجربة خارج العينة لنماذج بسيطة (أمس، متوسط 22 يوماً، HAR).
3) جدول أنظمة التقلب: تقلب الغد بعد أنظمة منخفضة/مرتفعة.
4) تحجيم المراكز: هل يُثبّت التقلب الذي يقيسه 22 يوماً مخاطر الاستثمار؟
5) ملف التقلب خلال ساعات اليوم.
"""
import argparse
import math


def mean(xs):
    return sum(xs) / len(xs)


def corr(a, b):
    ma, mb = mean(a), mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va and vb else 0.0


def ols(X, y):
    """انحدار خطي بالمعادلات الطبيعية (X تتضمن عمود الثابت)."""
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(k)] + [sum(r[i] * t for r, t in zip(X, y))]
         for i in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(A[r][i]))
        A[i], A[p] = A[p], A[i]
        for r in range(k):
            if r != i:
                f = A[r][i] / A[i][i]
                A[r] = [a - f * b for a, b in zip(A[r], A[i])]
    return [A[i][k] / A[i][i] for i in range(k)]


def features(lrv, t):
    """لقطات معروفة عند نهاية اليوم t: أمس، متوسط 5، متوسط 22 (على log التقلب)."""
    return lrv[t], mean(lrv[t - 4:t + 1]), mean(lrv[t - 21:t + 1])


def forecast_test(rv, train=0.7):
    lrv = [math.log(v) for v in rv]
    idx = list(range(21, len(lrv) - 1))
    cut = int(len(idx) * train)
    tr, te = idx[:cut], idx[cut:]
    y_tr, y_te = [lrv[t + 1] for t in tr], [lrv[t + 1] for t in te]
    base = mean(y_tr)
    sse0 = sum((y - base) ** 2 for y in y_te)

    def r2(pred):
        return 1 - sum((y - p) ** 2 for y, p in zip(y_te, pred)) / sse0

    out = {}
    out["أمس فقط (naive)"] = r2([lrv[t] for t in te])
    out["متوسط 22 يوماً"] = r2([features(lrv, t)[2] for t in te])
    X_tr = [[1.0, *features(lrv, t)] for t in tr]
    beta = ols(X_tr, y_tr)
    out["HAR (أمس+5+22، مُدرَّب)"] = r2([sum(b * x for b, x in zip(beta, [1.0, *features(lrv, t)]))
                                         for t in te])
    return out, beta


def regimes(rv, q=5):
    lrv = [math.log(v) for v in rv]
    rows = [(mean(rv[t - 21:t + 1]), rv[t + 1]) for t in range(21, len(rv) - 1)]
    rows.sort()
    size = len(rows) // q
    return [(mean([a for a, _ in rows[i * size:(i + 1) * size]]),
             mean([b for _, b in rows[i * size:(i + 1) * size]])) for i in range(q)]


def vol_targeting(dret, rv, block=63):
    """يقارن ثبات تذبذب العائد (على كتل ربعية) قبل/بعد التحجيم بتقلب 22 يوماً المعروف مسبقاً."""
    target = mean(rv)
    raw, scaled = [], []
    for t in range(22, len(dret)):
        w = min(3.0, target / mean(rv[t - 22:t]))  # يعتمد على ما قبل t فقط
        raw.append(dret[t])
        scaled.append(dret[t] * w)

    def block_sd(xs):
        sds = []
        for i in range(0, len(xs) - block + 1, block):
            b = xs[i:i + block]
            m = mean(b)
            sds.append(math.sqrt(sum((x - m) ** 2 for x in b) / (len(b) - 1)))
        return sds

    out = {}
    for name, xs in (("بدون تحجيم", raw), ("بتحجيم التقلب", scaled)):
        sds = block_sd(xs)
        m = mean(sds)
        out[name] = (m, math.sqrt(sum((s - m) ** 2 for s in sds) / (len(sds) - 1)) / m)
    return out


def hour_profile(hret, hours):
    g = {}
    for r, h in zip(hret, hours):
        if r is not None:
            g.setdefault(h, []).append(abs(r))
    return {h: mean(v) for h, v in sorted(g.items())}


def report(rv, dret, hret, hours, cost_bps):
    lrv = [math.log(v) for v in rv]
    print(f"أيام: {len(rv)} | تقلب يومي متوسط ≈ {mean(rv):.1f} نقطة أساس | تكلفة الصفقة ≈ {cost_bps:.2f}\n")

    print("--- 1) استمرارية التقلب (ارتباط log التقلب اليومي بنفسه بعد lag أيام) ---")
    for lag in (1, 2, 5, 10, 22, 63):
        print(f"lag {lag:>3}: {corr(lrv[lag:], lrv[:-lag]):+.3f}")

    print("\n--- 2) توقع تقلب الغد خارج العينة (R² مقابل متوسط التدريب؛ 0 = لا فائدة) ---")
    res, beta = forecast_test(rv)
    for k, v in res.items():
        print(f"{k:<26} R² = {v:+.3f}")
    print("معاملات HAR [ثابت, أمس, متوسط5, متوسط22]:", [round(b, 3) for b in beta])

    print("\n--- 3) أنظمة التقلب (خمسيات متوسط 22 يوماً → تقلب اليوم التالي) ---")
    rg = regimes(rv)
    for i, (a, b) in enumerate(rg, 1):
        print(f"خُمس {i}: تقلب 22 يوماً {a:5.1f} → تقلب الغد {b:5.1f}")
    print(f"نسبة أعلى خُمس إلى أدنى خُمس (غداً): {rg[-1][1] / rg[0][1]:.2f}x")

    print("\n--- 4) تحجيم المراكز (تذبذب العائد على كتل 63 يوماً؛ المعامل الأصغر = مخاطر أثبت) ---")
    for name, (m, cv) in vol_targeting(dret, rv).items():
        print(f"{name:<14} متوسط التذبذب {m:6.1f} | تباينه عبر الكتل (CV) {cv:.2f}")

    print("\n--- 5) متوسط |عائد الساعة| حسب ساعة UTC (نقطة أساس) ---")
    prof = hour_profile(hret, hours)
    mx = max(prof.values())
    for h, v in prof.items():
        print(f"{h:02d}: {v:5.2f} " + "#" * int(30 * v / mx))
    print(f"\nالتكلفة {cost_bps:.2f} نقطة مقابل أصغر حركة ساعية {min(prof.values()):.2f} وأكبرها {mx:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="bars_1h.parquet")
    a = ap.parse_args()
    import numpy as np
    import pandas as pd
    d = pd.read_parquet(a.cache)
    s = d["px"].copy()
    s.index = pd.DatetimeIndex(s.index)
    r = 1e4 * np.log(s).diff()
    gap = s.index.to_series().diff() > pd.Timedelta(hours=1)  # تجاهل الفجوات
    r[gap.values] = np.nan
    day = r.groupby(s.index.date)
    ok = day.count() >= 12
    rv = np.sqrt((r ** 2).groupby(s.index.date).sum())[ok]
    dret = r.groupby(s.index.date).sum()[ok]
    keep = rv > 0
    print(f"{s.index[0].date()} -> {s.index[-1].date()}")
    hret = [None if np.isnan(x) else x for x in r.tolist()]
    report(rv[keep].tolist(), dret[keep].tolist(), hret, s.index.hour.tolist(),
           float(d["spread"].iloc[0]) * 1e4)


if __name__ == "__main__":
    main()

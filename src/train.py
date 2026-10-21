"""モデルを学習して model/ に保存する（元の CSV は非公開のため、リポジトリには入れない）。
使い方: python src/train.py private/2026-10-15_videos.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
sys.path.insert(0, str(Path(__file__).parent))
from features import build_from_studio_csv, FEATURES

ROOT = Path(__file__).resolve().parents[1]
from train_params import P  # パラメータは export_cv.py と共有
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

X, y, v = build_from_studio_csv(sys.argv[1])
rows, cover, resid = [], [], []   # resid: (fold, Shorts, 実際−予測)
for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=5).split(X)):
    pred = lgb.LGBMRegressor(**P).fit(X.loc[tr], y[tr]).predict(X.loc[te])
    rows.append({"median": rmse(y[te], np.full(len(te), y[tr].median())),
                 "ols": rmse(y[te], LinearRegression().fit(X.loc[tr], y[tr]).predict(X.loc[te])),
                 "lgbm": rmse(y[te], pred), "n_test": len(te)})
    r = pd.DataFrame({"fold": k, "Shorts": X.loc[te, "Shorts"].values, "r": y[te].values - pred})
    # 区間の当たり方：それより前の fold の残差だけで 10%/90% 点を決め、この fold で確かめる（未来の情報を使わない）
    if k >= 2:
        past = pd.concat(resid)
        for s_, g in r.groupby("Shorts"):
            q = past[past.Shorts == s_]["r"].quantile([0.1, 0.9]).values
            cover.extend(((g["r"] >= q[0]) & (g["r"] <= q[1])).tolist())
    resid.append(r)
resid = pd.concat(resid)
cv = pd.DataFrame(rows)
w = cv["n_test"] / cv["n_test"].sum()
summary = {k: round(float((cv[k] * w).sum()), 3) for k in ["median", "ols", "lgbm"]}
summary["interval_80_coverage"] = round(float(np.mean(cover)), 3)
print(cv.round(3)); print(summary)

(ROOT / "model").mkdir(exist_ok=True)
lgb.LGBMRegressor(**P).fit(X, y).booster_.save_model(str(ROOT / "model" / "point.txt"))

meta = {
    "features": FEATURES, "n_videos": int(len(X)), "n_shorts": int(X["Shorts"].sum()),
    "period": [str(v["公開日"].min().date()), str(v["公開日"].max().date())],
    "cv_rmse_log": summary,
    "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES},
    "median_views": {"Shorts": float(np.expm1(y[X.Shorts == 1].median())), "長尺": float(np.expm1(y[X.Shorts == 0].median()))},
    "max_days_since_start": int(X["開始からの日数"].max()),
    "residual_q10_q90": {("Shorts" if k else "長尺"): [round(float(x), 4) for x in g["r"].quantile([0.1, 0.9]).values]
                         for k, g in resid.groupby("Shorts")},
}
meta["medians_by_type"] = {}
for s_, name in ((1, "Shorts"), (0, "長尺")):
    g = X[X.Shorts == s_]
    meta["medians_by_type"][name] = {c: float(g[c].median()) for c in FEATURES}
    meta["medians_by_type"][name]["length_sec_median"] = float(np.expm1(g["log長さ"].median()))
json.dump(meta, open(ROOT / "model" / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(meta, ensure_ascii=False)[:600])

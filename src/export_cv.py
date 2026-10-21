"""時系列 CV の予測（各 fold のテスト部分）を書き出す。アプリの「精度」ページ用。
train.py と同じ分け方・同じパラメータなので、数値は train.py の CV と一致する。
出すのは log(1+視聴回数) の実測・予測と fold・種類だけ（タイトルや動画 ID は出さない）。
使い方: python src/export_cv.py private/2026-10-15_videos.csv
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
sys.path.insert(0, str(Path(__file__).parent))
from features import build_from_studio_csv
from train_params import P

ROOT = Path(__file__).resolve().parents[1]
X, y, v = build_from_studio_csv(sys.argv[1])
out = []
for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=5).split(X)):
    pred = lgb.LGBMRegressor(**P).fit(X.loc[tr], y[tr]).predict(X.loc[te])
    ols = LinearRegression().fit(X.loc[tr], y[tr]).predict(X.loc[te])
    out.append(pd.DataFrame({"fold": k + 1, "種類": np.where(X.loc[te, "Shorts"].values == 1, "Shorts", "長尺"),
                             "実測_log": y[te].values, "予測_log": pred, "重回帰_log": ols,
                             "中央値_log": float(y[tr].median())}))
df = pd.concat(out).round(3)
df.to_csv(ROOT / "model" / "cv_predictions.csv", index=False, encoding="utf-8")
rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
print(df.groupby("fold").apply(lambda g: pd.Series({"n": len(g), "lgbm": rm(g.実測_log, g.予測_log), "ols": rm(g.実測_log, g.重回帰_log), "median": rm(g.実測_log, g.中央値_log)})).round(3))
print("全体 lgbm", round(rm(df.実測_log, df.予測_log), 3), "n", len(df))

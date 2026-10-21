"""公開時点でわかる情報だけで特徴量を作る（youtube-data-mart の e_06 と同じ定義）。"""
import re
import numpy as np
import pandas as pd

FEATURES = ["Shorts", "log長さ", "曜日", "土日", "前の投稿からの日数", "同じ種類の前の投稿からの日数",
            "同じ日の投稿本数", "タイトル文字数", "絵文字あり", "#shortsあり", "開始からの日数"]
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def build_from_studio_csv(path, cutoff="2026-10-15"):
    v = pd.read_csv(path).iloc[1:].copy()           # 1行目は合計行
    v["公開日"] = pd.to_datetime(v["動画公開時刻"], format="%b %d, %Y")
    title = v["動画のタイトル"].astype(str)
    v = v[~title.str.contains("LIVE") & (v["長さ"] > 0)].copy()
    v["Shorts"] = (v["長さ"] <= 60).astype(int)
    v = v.sort_values(["公開日", "コンテンツ"]).reset_index(drop=True)
    title = v["動画のタイトル"].astype(str)
    X = pd.DataFrame({
        "Shorts": v["Shorts"],
        "log長さ": np.log1p(v["長さ"]),
        "曜日": v["公開日"].dt.dayofweek,
        "土日": (v["公開日"].dt.dayofweek >= 5).astype(int),
        "前の投稿からの日数": v["公開日"].diff().dt.days.fillna(0),
        "同じ種類の前の投稿からの日数": v.groupby("Shorts")["公開日"].diff().dt.days.fillna(0),
        "同じ日の投稿本数": v.groupby("公開日")["コンテンツ"].transform("size"),
        "タイトル文字数": title.str.len(),
        "絵文字あり": title.apply(lambda s: int(bool(EMOJI.search(s)))),
        "#shortsあり": title.str.contains("#shorts", case=False).astype(int),
        "開始からの日数": (v["公開日"] - v["公開日"].min()).dt.days,
    })
    y = np.log1p(v["視聴回数"].astype(float))
    keep = ((v["公開日"] <= pd.Timestamp(cutoff) - pd.Timedelta(days=7)) & v["視聴回数"].notna()).values
    return X[keep].reset_index(drop=True), y[keep].reset_index(drop=True), v[keep].reset_index(drop=True)


def from_inputs(shorts, length_sec, weekday, gap_all, gap_same, same_day, title, days_since_start):
    return pd.DataFrame([{
        "Shorts": int(shorts), "log長さ": float(np.log1p(length_sec)), "曜日": int(weekday),
        "土日": int(weekday >= 5), "前の投稿からの日数": float(gap_all), "同じ種類の前の投稿からの日数": float(gap_same),
        "同じ日の投稿本数": int(same_day), "タイトル文字数": len(title), "絵文字あり": int(bool(EMOJI.search(title))),
        "#shortsあり": int("#shorts" in title.lower()), "開始からの日数": int(days_since_start),
    }])[FEATURES]

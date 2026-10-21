"""公開前の条件から、YouTube 動画の再生数を予測するアプリ（LightGBM・寄与の分解つき）。
自分のチャンネル（2026年7〜10月・299本）で学習したモデル。元データは同梱していない。
"""
import json
from pathlib import Path

import altair as alt
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st

from src.features import FEATURES

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def load():
    booster = lgb.Booster(model_file=str(ROOT / "model" / "point.txt"))
    meta = json.load(open(ROOT / "model" / "meta.json", encoding="utf-8"))
    return booster, meta


booster, meta = load()
WEEK = ["月", "火", "水", "木", "金", "土", "日"]

st.title("公開前の条件から、再生数を予測する")
cv = meta["cv_rmse_log"]
st.markdown(f"""
- **何を予測するか**：動画を出す前に、公開後の再生数（Shorts はほぼ公開7日の再生数、長尺は累計）がどれくらいになりそうか
- **使ったデータ**：自分の YouTube チャンネルの書き出し（{meta['period'][0]}〜{meta['period'][1]} 公開・{meta['n_videos']}本）。公開後の数値（クリック率など）は使っていない
- **モデルの精度**：LightGBM。公開日順の時系列 CV で誤差（log の RMSE）{cv['lgbm']}。中央値で当てた場合（{cv['median']}）より約{(1-cv['lgbm']/cv['median'])*100:.0f}%小さく、重回帰（{cv['ols']}）とほぼ同じ
- **限界**：予測であって因果ではない。Shorts 同士の差は公開前の情報ではほとんど説明できない。1チャンネル・約3か月のデータ
""")

with st.sidebar:
    st.header("動画の条件")
    st.caption("よく変える4つだけを動かせます。ほかの条件（投稿の間隔・1日の本数・公開の時期など）は、学習データの中央値に固定しています。")
    kind = st.radio("種類", ["Shorts", "長尺"], horizontal=True)
    med = meta["medians_by_type"][kind]
    if kind == "Shorts":
        length = st.slider("長さ（秒）", 5, 60, int(round(med["length_sec_median"])))
    else:
        length = st.slider("長さ（時間）", 0.5, 12.0, float(round(med["length_sec_median"] / 3600 * 2) / 2), step=0.5) * 3600
    weekday = st.select_slider("公開の曜日", options=list(range(7)), value=int(med["曜日"]), format_func=lambda i: WEEK[i])
    title_len = st.slider("タイトルの文字数", 10, 100, int(med["タイトル文字数"]))

row = {c: med[c] for c in FEATURES}
row.update({"Shorts": int(kind == "Shorts"), "log長さ": float(np.log1p(length)), "曜日": weekday, "土日": int(weekday >= 5),
            "タイトル文字数": title_len, "#shortsあり": int(kind == "Shorts")})
X = pd.DataFrame([row])[FEATURES]
pred_log = float(booster.predict(X)[0])
q10, q90 = meta["residual_q10_q90"][kind]
point, lo, hi = np.expm1(pred_log), np.expm1(pred_log + q10), np.expm1(pred_log + q90)

c1, c2 = st.columns([1, 1])
c1.metric("予測（公開後の再生数）", f"{point:,.0f} 回")
c2.metric("おおよその範囲", f"{max(lo, 0):,.0f} 〜 {hi:,.0f} 回")
st.caption(
    f"範囲は、過去の予測の外れ方（10%〜90%の幅）から作っています。時期を追った検証では、実際の値がこの範囲に入ったのは約{meta['cv_rmse_log']['interval_80_coverage']*100:.0f}%で、"
    "目標の80%より少なめです。目安として見てください。"
    f"　参考：学習データの中央値は Shorts {meta['median_views']['Shorts']:,.0f} 回、長尺 {meta['median_views']['長尺']:,.0f} 回。"
)

# --- 寄与の分解（LightGBM の pred_contrib ＝ TreeSHAP と同じ値） ---
contrib = booster.predict(X, pred_contrib=True)[0]
base, parts = contrib[-1], pd.Series(contrib[:-1], index=FEATURES)
labels = {"Shorts": "種類（Shorts）", "log長さ": "長さ", "曜日": "曜日", "土日": "土日", "前の投稿からの日数": "前の投稿からの日数",
          "同じ種類の前の投稿からの日数": "同じ種類の間隔", "同じ日の投稿本数": "同じ日の本数", "タイトル文字数": "タイトルの長さ",
          "絵文字あり": "絵文字", "#shortsあり": "#shorts", "開始からの日数": "公開の時期"}
df = pd.DataFrame({"条件": [labels[f] for f in FEATURES], "寄与": parts.values})
df["倍率"] = np.exp(df["寄与"])
df = df.reindex(df["寄与"].abs().sort_values(ascending=False).index)

st.subheader("何が予測を上げ下げしたか")
st.caption(f"平均的な動画（約{np.expm1(base):,.0f}回）から、各条件で何倍になったか。1より大きいと上げ、小さいと下げています。")
top = df.head(6).assign(向き=lambda d: np.where(d["寄与"] >= 0, "上げた", "下げた"), 表示=lambda d: d["倍率"].map(lambda x: f"×{x:.2f}"))
chart = alt.Chart(top).mark_bar(cornerRadiusEnd=4, height=18).encode(
    x=alt.X("倍率:Q", scale=alt.Scale(type="log", domain=[min(0.8, top["倍率"].min() * 0.9), max(1.25, top["倍率"].max() * 1.1)]),
            title="倍率（1 ＝ 変化なし、対数目盛）"),
    x2=alt.datum(1),
    y=alt.Y("条件:N", sort=list(top["条件"]), title=None, axis=alt.Axis(labelOverlap=False, labelFontSize=13, labelLimit=200)),
    color=alt.Color("向き:N", scale=alt.Scale(domain=["上げた", "下げた"], range=["#2a78d6", "#eb6834"]), legend=alt.Legend(title=None, orient="top")),
    tooltip=["条件", "表示"],
).properties(height=280)
st.altair_chart(chart, use_container_width=True)
st.dataframe(df.assign(倍率=df["倍率"].map(lambda x: f"×{x:.2f}"))[["条件", "倍率"]], hide_index=True, use_container_width=True)

with st.expander("SHAP のウォーターフォール図（log の値で、平均からこの予測までの積み上げ）"):
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "Hiragino Sans", "sans-serif"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    order = list(df.sort_values("寄与", key=abs, ascending=False).itertuples())
    cur = base
    for i, r in enumerate(order):
        color = "#2a78d6" if r.寄与 >= 0 else "#eb6834"
        ax.barh(len(order) - 1 - i, r.寄与, left=cur, color=color, height=0.6)
        ax.text(cur + r.寄与 + (0.01 if r.寄与 >= 0 else -0.01), len(order) - 1 - i, f"{r.寄与:+.2f}",
                va="center", ha="left" if r.寄与 >= 0 else "right", fontsize=8, color="#52514e")
        cur += r.寄与
    ax.set_yticks(range(len(order)), [r.条件 for r in order][::-1], fontsize=9)
    ax.axvline(base, color="#a3a29c", lw=1, ls="--"); ax.axvline(cur, color="#0b0b0b", lw=1)
    ax.set_xlabel(f"log(1＋再生数)　平均 {base:.2f} → 予測 {cur:.2f}（約{np.expm1(cur):,.0f}回）")
    pts = np.cumsum([base] + [r.寄与 for r in order]); pad = (pts.max() - pts.min()) * 0.15
    ax.set_xlim(pts.min() - pad, pts.max() + pad)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    st.pyplot(fig)
    st.caption("値は LightGBM の pred_contrib（TreeSHAP と同じ計算）。青は予測を上げた条件、オレンジは下げた条件。")

with st.expander("このモデルについて（精度と限界）"):
    cv = meta["cv_rmse_log"]
    st.markdown(f"""
- **精度**：公開日順に5分割した時系列の検証で、誤差（log(1＋再生数) の RMSE）は LightGBM **{cv['lgbm']}**、重回帰 {cv['ols']}、中央値で当てた場合 {cv['median']}。中央値より約{(1-cv['lgbm']/cv['median'])*100:.0f}%小さいが、重回帰とほぼ同じ。
- **効いているのは種類と長さと時期**。タイトルの長さ・絵文字・曜日・投稿間隔の寄与は小さい。
- **Shorts 同士の差は、公開前の情報ではほぼ説明できない**（Shorts だけで比べると中央値に勝てない）。差はサムネイル・冒頭・おすすめへの乗り方など、公開後に決まる部分が大きい。
- **これは予測で、因果ではない**。「曜日を変えれば再生数が変わる」とは言えない。打ち手の効果は、別に A/B の実験で確かめている。
- **一つのチャンネル・約3か月**のデータなので、ほかのチャンネルや先の時期にはそのまま当てはまらない。
""")
st.info("どのくらい当たるかは、左のメニューの「精度」ページで、実測と予測の散布図と一緒に見られます。")
st.caption("コード：github.com/takayuki-0127/youtube-views-predictor ／ 分析の全体：github.com/takayuki-0127/youtube-data-mart")

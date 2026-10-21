"""2ページ目：このモデルはどのくらい当たるか（時系列 CV の結果）"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

meta = json.load(open(ROOT / "model" / "meta.json", encoding="utf-8"))
cv = meta["cv_rmse_log"]
df = pd.read_csv(ROOT / "model" / "cv_predictions.csv")
df["実測"] = np.expm1(df["実測_log"])
df["予測"] = np.expm1(df["予測_log"])
err = (df["予測_log"] - df["実測_log"]).abs()

st.title("どのくらい当たるか")
st.markdown(
    f"公開日順に5つに分け、**それより前の動画だけで学習して、次の期間の動画を予測する**（時系列の交差検証）。"
    f"未来の情報を使わないので、実際に新しい動画を予測するときに近い当たり方になる。対象は {meta['period'][0]}〜{meta['period'][1]} 公開の {meta['n_videos']} 本のうち、予測した {len(df)} 本。"
)

c1, c2, c3 = st.columns(3)
c1.metric("2倍以内に当たった割合", f"{(err <= np.log(2)).mean() * 100:.0f}%")
c2.metric("3倍以内に当たった割合", f"{(err <= np.log(3)).mean() * 100:.0f}%")
c3.metric("おおよその範囲に入った割合", f"{cv['interval_80_coverage'] * 100:.0f}%", help="1ページ目の「おおよその範囲」（10%〜90%）に実測が入った割合。目標は80%")

st.subheader("誤差の比較（log の RMSE、小さいほどよい）")
cmp = pd.DataFrame({"方法": ["中央値で当てる（何もしない）", "重回帰", "LightGBM（このアプリ）"],
                    "誤差": [cv["median"], cv["ols"], cv["lgbm"]]})
bar = alt.Chart(cmp).mark_bar(cornerRadiusEnd=4, height=22).encode(
    x=alt.X("誤差:Q", title="RMSE（log(1＋再生数)）", scale=alt.Scale(domain=[0, max(cmp["誤差"]) * 1.15])),
    y=alt.Y("方法:N", sort=list(cmp["方法"]), title=None, axis=alt.Axis(labelLimit=220, labelFontSize=13)),
    color=alt.condition(alt.datum.方法 == "LightGBM（このアプリ）", alt.value("#2a78d6"), alt.value("#a3a29c")),
    tooltip=["方法", alt.Tooltip("誤差:Q", format=".3f")],
).properties(height=150)
st.altair_chart(bar + bar.mark_text(align="left", dx=4).encode(text=alt.Text("誤差:Q", format=".3f")), use_container_width=True)
st.caption(f"中央値で当てるより約{(1 - cv['lgbm'] / cv['median']) * 100:.0f}%小さい。重回帰とはほぼ同じで、木のモデルにした上積みは小さい。")

st.subheader("実測と予測（1点が1本）")
kinds = st.multiselect("表示する種類", ["Shorts", "長尺"], default=["Shorts", "長尺"])
d = df[df["種類"].isin(kinds)]
lo, hi = 1, max(df["実測"].max(), df["予測"].max()) * 1.2
line = alt.Chart(pd.DataFrame({"x": [lo, hi]})).mark_line(color="#52514e", strokeDash=[4, 4]).encode(x="x:Q", y="x:Q")
band = alt.Chart(pd.DataFrame({"x": [lo, hi], "lo": [lo / 2, hi / 2], "hi": [lo * 2, hi * 2]})).mark_area(opacity=0.12, color="#2a78d6").encode(x="x:Q", y="lo:Q", y2="hi:Q")
pts = alt.Chart(d.assign(実測1=d["実測"].clip(lower=1), 予測1=d["予測"].clip(lower=1))).mark_circle(size=42, opacity=0.7).encode(
    x=alt.X("実測1:Q", scale=alt.Scale(type="log", domain=[lo, hi]), title="実際の再生数（対数目盛）"),
    y=alt.Y("予測1:Q", scale=alt.Scale(type="log", domain=[lo, hi]), title="予測した再生数（対数目盛）"),
    color=alt.Color("種類:N", scale=alt.Scale(domain=["Shorts", "長尺"], range=["#2a78d6", "#eb6834"]), legend=alt.Legend(orient="top", title=None)),
    tooltip=[alt.Tooltip("実測:Q", format=",.0f"), alt.Tooltip("予測:Q", format=",.0f"), "種類", "fold"],
).properties(height=420)
st.altair_chart(band + line + pts, use_container_width=True)
st.caption("点線の上なら完全に当たり。薄い帯の中が「2倍以内」。0回の動画は1回として表示。")

rows = []
for k, g in df.groupby("種類"):
    e = (g["予測_log"] - g["実測_log"]).abs()
    rows.append({"種類": k, "本数": len(g), "2倍以内": f"{(e <= np.log(2)).mean() * 100:.0f}%",
                 "同じ種類の中での相関（log）": round(float(np.corrcoef(g["実測_log"], g["予測_log"])[0, 1]), 2)})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.subheader("読み方")
st.markdown("""
- **当たっているのは「Shorts か長尺か」の差がほとんど**。全体では右上がりに並ぶが、同じ種類の中だけで見ると、実測と予測の相関はほぼ0。
- つまり、このモデルで言えるのは「Shorts ならこのくらい、長尺ならこのくらい」という**水準の見当**まで。同じ Shorts の中で「どれが伸びるか」は、公開前の情報ではほぼ分からない。
- 伸びるかどうかは、サムネイル・冒頭の数秒・おすすめへの乗り方など公開後に決まる部分が大きい。そこは予測ではなく、A/B の実験で確かめている。
""")
st.caption("データ：自分の YouTube チャンネルの書き出し。元の CSV は公開していない（このページの点は log の実測・予測と種類・fold だけ）。")

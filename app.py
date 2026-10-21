"""公開前の条件から YouTube 動画の再生数を予測するアプリ（入口）。
ページ：予測（views/predict.py）／精度（views/accuracy.py）
"""
import streamlit as st

st.set_page_config(page_title="再生数の予測（公開前）", page_icon="📈", layout="centered")
nav = st.navigation([
    st.Page("views/predict.py", title="予測", icon="📈", default=True),
    st.Page("views/accuracy.py", title="精度：どのくらい当たるか", icon="🎯", url_path="accuracy"),
])
nav.run()

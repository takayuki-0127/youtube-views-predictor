# youtube-views-predictor：手元で動かす手順と、無料枠で公開する手順（2026-10-21）

## 手元（Windows）で動かす
1. コマンドプロンプトで `cd %USERPROFILE%\Desktop\youtube-views-predictor`
2. （初回だけ）`python -m venv .venv` → `.venv\Scripts\activate` → `pip install -r requirements.txt`
3. `streamlit run app.py` → ブラウザで http://localhost:8501 が開く
4. 左の「動画の条件」を変えると、予測・おおよその範囲・「何が予測を上げ下げしたか」・**SHAP のウォーターフォール図**（開閉式）が変わる
- 動かすのに要るのは `model/point.txt` と `model/meta.json` だけ。元の CSV は要らない
- 学習し直すとき：`python src/train.py <YouTube Studio の動画一覧 CSV>`

## 画面の中身（いちばん小さい構成）
| 部品 | 中身 |
|---|---|
| 入力 | 種類・長さ・曜日・タイトル・投稿の間隔など（公開前にわかる情報だけ） |
| 予測 | LightGBM（`model/point.txt`）の予測を回数に戻したもの＋過去の外れ方から作った範囲 |
| 寄与 | `booster.predict(X, pred_contrib=True)`＝TreeSHAP と同じ値。横棒（倍率）と、ウォーターフォール図（log の値） |
| 説明 | 精度（時系列 CV の RMSE）と限界 |

shap ライブラリは使っていない（LightGBM の pred_contrib で同じ値が出るので、依存を減らした）。

## 無料枠（Streamlit Community Cloud）で公開する手順 ※まだやっていない
1. https://share.streamlit.io を開き、GitHub アカウントでサインインする（初回は GitHub への連携の許可を求められる。リポジトリの読み取りを許可する）
2. 「Create app」→「Deploy a public app from GitHub」
3. Repository：`takayuki-0127/youtube-views-predictor`／Branch：`main`／Main file path：`app.py`
4. App URL を決める（例：`youtube-views-predictor.streamlit.app`）
5. 「Advanced settings」で Python のバージョンを 3.12 にする
6. 「Deploy」→ 数分で起動。`packages.txt` の `libgomp1`（LightGBM に必要）と `fonts-noto-cjk`（図の日本語）も自動で入る
7. 公開できたら、README の先頭とプロフィール README にアプリの URL を足す

注意
- 無料枠は公開リポジトリのみ。しばらく使われないとアプリは休止し、次に開いたときに数十秒かけて起き上がる
- 元の CSV はリポジトリに入れていないので、公開しても自分のチャンネルの生データは出ない（出るのは学習済みモデルと集計値だけ）

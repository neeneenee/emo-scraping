
## 📘 日本語ポスト分析アプリ

日本語テキストまたは URL を入力すると、自動で本文を取得し、  
- Positive/Negative 感情分析  
- Plutchik 8軸感情分析  
- 高品質要約 + 自動校正  

を行う Streamlit アプリです。

## 📦 インストール

git clone https://github.com/neeneenee/emo-scraping.git

cd emo-scraping

python -m venv env

source env/bin/activate    # Windows: env\Scripts\activate

pip install -r requirements.txt



## 🚀 実行

streamlit run app.py

ブラウザが自動で立ち上がります。

## requirements

streamlit>=1.46.0

transformers>=4.52.4

sentencepiece>=0.2.0

fugashi>=1.5.1

unidic-lite>=1.0.8

torch>=2.7.1


pandas>=2.3.0

plotly>=6.1.2

requests>=2.32.4

beautifulsoup4>=4.13.4

lxml>=5.4.0

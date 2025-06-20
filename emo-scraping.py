# 一度だけ実行
# pip install transformers sentencepiece fugashi unidic-lite torch pandas plotly requests beautifulsoup4 lxml

import streamlit as st
import torch
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5Tokenizer,
    AutoModelForSeq2SeqLM,
)
import plotly.express as px

# ─── モデル読み込み ───────────────────────────────────────────
@st.cache_resource
def load_models():
    # P/N 感情分析
    s_name   = "jarvisx17/japanese-sentiment-analysis"
    tok_s    = AutoTokenizer.from_pretrained(s_name)
    model_s  = AutoModelForSequenceClassification.from_pretrained(s_name)
    # 要約モデル（legacy=False 指定）
    t5_name  = "Zolyer/ja-t5-base-summary"
    tok_t5   = T5Tokenizer.from_pretrained(t5_name, use_fast=False, legacy=False)
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(t5_name)
    # 8軸感情（WRIME fine-tune）
    e_name   = "patrickramos/bert-base-japanese-v2-wrime-fine-tune"
    tok_e    = AutoTokenizer.from_pretrained(e_name)
    model_e  = AutoModelForSequenceClassification.from_pretrained(e_name)
    # 校正用にも同じ T5（legacy=False）
    cor_tok   = tok_t5
    cor_model = model_t5
    return (tok_s, model_s), (tok_t5, model_t5), (tok_e, model_e), (cor_tok, cor_model)

(sent_tok, sent_model), (sum_tok, sum_model), (emo_tok, emo_model), (cor_tok, cor_model) = load_models()

# ─── URL→本文抽出 ────────────────────────────────────────────
def fetch_text(url: str) -> str:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    return "\n".join(p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip())

# ─── 感情分析 (P/N) ───────────────────────────────────────────
def analyze_sentiment(text: str):
    # 必ず max_length=512 で truncate & pad
    inputs = sent_tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    with torch.no_grad():
        logits = sent_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    score, idx = torch.max(probs, dim=0)
    label = sent_model.config.id2label[idx.item()]
    return label, score.item()

# ─── Plutchik の 8 感情分析 ──────────────────────────────────
def analyze_plutchik8(text: str):
    inputs = emo_tok(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    with torch.no_grad():
        raw = emo_model(**inputs).logits[0][:8].tolist()
    labels = ["歓喜", "悲哀", "期待", "驚愕", "憤怒", "恐怖", "嫌悪", "信頼"]
    return {lab: round(val, 3) for lab, val in zip(labels, raw)}

# ─── 要約 (自動トークン長設定) ─────────────────────────────────
def summarize(text: str) -> str:
    if len(text) < 20:
        return text

    prefixed = "要約: " + text

    # ① 入力長計測（truncate だけ）
    len_inputs = sum_tok(
        prefixed,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    token_len = len_inputs["input_ids"].shape[1]

    # ② 本番トークナイズ（512 固定で pad & truncate）
    inputs = sum_tok(
        prefixed,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )

    # ③ max/min 長を自動設定
    max_len = min(int(token_len * 0.7) + 10, 200)
    min_len = max(int(token_len * 0.3), 10)
    if min_len > max_len:
        min_len = max_len

    summary_ids = sum_model.generate(
        **inputs,
        max_length=max_len,
        min_length=min_len,
        num_beams=6,
        early_stopping=True,
        length_penalty=1.0,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
    )
    return sum_tok.decode(summary_ids[0], skip_special_tokens=True)

# ─── 校正 (Grammar Correction) ─────────────────────────────────
def correct_text(text: str) -> str:
    pref = "校正: " + text

    inputs = cor_tok(
        pref,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    max_len = min(inputs["input_ids"].shape[1] + 50, 512)
    with torch.no_grad():
        ids = cor_model.generate(
            **inputs,
            max_length=max_len,
            num_beams=4,
            early_stopping=True,
        )
    return cor_tok.decode(ids[0], skip_special_tokens=True)

# ─── Streamlit UI ────────────────────────────────────────────
st.markdown("<h2 style='font-size:22px;'>📘 日本語ポスト分析アプリ</h2>", unsafe_allow_html=True)
st.write("URL または テキスト を入力し、「分析実行」を押してください。")

raw = st.text_input("🔹 URL or テキスト", "")
if st.button("分析実行") and raw:
    if raw.startswith("http"):
        with st.spinner("Webページを取得中…"):
            try:
                text = fetch_text(raw)
                st.expander("本文プレビュー").write(text)
            except Exception as e:
                st.error(f"取得エラー: {e}")
                st.stop()
    else:
        text = raw

    with st.spinner("分析中…"):
        label             = analyze_sentiment(text)[0]
        score             = analyze_sentiment(text)[1]
        plutchik8         = analyze_plutchik8(text)
        summary           = summarize(text)
        corrected_summary = correct_text(summary)

    # デフォルトで 8軸を表示
    col1, col2 = st.columns([1, 2])
    with col1:
        df = (
            pd.DataFrame.from_dict(plutchik8, orient="index", columns=["score"])
              .sort_values("score", ascending=False)
              .reset_index()
              .rename(columns={"index": "感情"})
        )
        height = (2 + len(df)) * 60
        fig = px.bar(
            df, x="感情", y="score",
            title="8軸 Plutchik 感情分析",
            labels={"score": "強度"},
            height=height
        )
        fig.update_layout(xaxis={'categoryorder':'total descending'},
                          font_family="Meiryo")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🧠 感情分析結果 (P/N)")
        st.write(f"ラベル: **{label}**  ／  スコア: **{score:.4f}**")
        st.subheader("🔷 8軸 Plutchik 感情結果")
        st.table(plutchik8)
        st.subheader("📝 要約結果 (校正済)")
        st.write(corrected_summary)

# llm.py
import os
import time
import torch
import streamlit as st
from transformers import pipeline
from huggingface_hub import login as hf_login
from config import MODEL_NAME

@st.cache_resource(show_spinner=False)
def load_model():
    """LLMモデルをロードし、キャッシュして再利用する"""
    # Hugging Face トークン取得
    hf_token = st.secrets.get("huggingface", {}).get("token")
    if not hf_token:
        raise ValueError("Hugging Face のトークンが設定されていません。.streamlit/secrets.toml を確認してください。")

    # 認証 (transformers の download に使う)
    hf_login(token=hf_token)

    # デバイス設定
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    st.info(f"Using device: {'cuda' if device >= 0 else 'cpu'}")

    # パイプライン初期化 (アクセストークンは hf_login で済ませる)
    pipe = pipeline(
        task="text-generation",
        model=MODEL_NAME,
        device=device,
        torch_dtype=torch_dtype,
    )

    st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
    return pipe


def generate_response(pipe, user_question: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9):
    """LLMを使用して質問に対する回答を生成し、応答時間を返す"""
    if pipe is None:
        return "モデルがロードされていないため、回答を生成できません。", 0.0

    start_time = time.time()
    try:
        outputs = pipe(
            user_question,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            return_full_text=False
        )
        text = outputs[0].get("generated_text", "").strip()
        if not text:
            st.warning("モデルからの応答が空でした。")
            text = "回答の抽出に失敗しました。"
        elapsed = time.time() - start_time
        return text, elapsed

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        return f"エラーが発生しました: {e}", 0.0

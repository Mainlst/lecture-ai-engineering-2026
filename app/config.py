"""講座全体で共通の設定。

APIキーやモデル名はコードに書かず、環境変数から読み込む。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# リポジトリ直下の .env を読み込む。既に設定済みの環境変数は上書きしない。
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# AIE_PROVIDER を省略した場合、利用可能なキーからプロバイダーを選ぶ。
# Gemini の既存環境はそのまま動作し、OpenAI キーだけの環境では OpenAI を使う。
PROVIDER = os.getenv(
    "AIE_PROVIDER",
    (
        "openai"
        if os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY")
        else "google"
    ),
).lower()
DEFAULT_MODELS = {"google": "gemini-3.5-flash-lite", "openai": "gpt-6-luna"}
if PROVIDER not in DEFAULT_MODELS:
    raise SystemExit("AIE_PROVIDER は 'google' または 'openai' を指定してください。")
MODEL = os.getenv("AIE_MODEL", DEFAULT_MODELS[PROVIDER])


def api_error_message(status: int, detail: str = "") -> str:
    """HTTPステータスコードを、受講生向けの日本語の説明にする（ask.py と agent.py で共用）。"""
    if status == 429:
        return "レート制限に達しました。1分ほど待ってから再実行してください。"
    if status == 503:
        return "モデルが混み合っています（503）。少し待ってから再実行してください。"
    if status in (400, 401, 403, 404):
        return f"APIキーかモデル名に問題があります（{status}）: {detail}"
    return f"APIエラー（{status}）: {detail}"


def get_api_key(provider: str = PROVIDER) -> str:
    """選択したプロバイダーの API キーを返す。"""
    env_name = "OPENAI_API_KEY" if provider == "openai" else "GEMINI_API_KEY"
    key = os.getenv(env_name)
    if not key:
        raise SystemExit(
            f"環境変数 {env_name} が見つかりません。\n"
            f"リポジトリ直下の .env に {env_name}=... を設定するか、環境変数を設定してください。"
        )
    return key

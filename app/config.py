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

# 第5回: 1ページを1チャンクとして検索する。索引はscripts/build_rag_index.pyで作成する。
ROOT = Path(__file__).resolve().parent.parent
RAG_SOURCE = ROOT / "data/rag_data/lectures.json"
RAG_TREE = ROOT / "data/rag_data/tree.json"
RAG_GRAPH = ROOT / "data/rag_data/graph.json"
RAG_DB = Path(os.getenv("AIE_RAG_DB", str(ROOT / "data/rag_data/index.sqlite3")))
RAG_DOCUMENT_IDS = ("lec01-main", "lec02-main", "lec03-main")
RAG_TOP_K = 10
GRAPH_START_K = 3
GRAPH_HOPS = 2
GRAPH_MAX_RESULTS = 10
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
# 2,048トークンの上限に対し、タスク指定等の分を残す。超過時は切り捨てず停止する。
EMBEDDING_MAX_TOKENS = 2000


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


def build_model():
    """選択したプロバイダーの生成モデルを作る。"""
    if PROVIDER == "openai":
        from pydantic_ai.models.openai import OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIResponsesModel(
            MODEL, provider=OpenAIProvider(api_key=get_api_key("openai"))
        )

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    return GoogleModel(MODEL, provider=GoogleProvider(api_key=get_api_key("google")))

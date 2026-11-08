"""ADK エージェントを、進行状況を表示しながら対話形式で実行する。"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from google.adk.cli.utils.envs import load_dotenv_for_agent
from google.adk.runners import InMemoryRunner
from google.genai import types

from .agent import root_agent

APP_NAME = "learning_agent"
USER_ID = "local_user"
PROGRESS_INTERVAL = 5
DEFAULT_TIMEOUT = 60


async def show_progress(done: asyncio.Event, started: float) -> None:
    """応答を待っている間、定期的に経過時間を表示する。"""
    while True:
        try:
            await asyncio.wait_for(done.wait(), timeout=PROGRESS_INTERVAL)
            return
        except TimeoutError:
            elapsed = int(time.monotonic() - started)
            print(f"[待機中] 応答を待っています（{elapsed}秒経過）", flush=True)


async def ask(
    runner: InMemoryRunner, session_id: str, question: str, timeout: float
) -> None:
    done = asyncio.Event()
    started = time.monotonic()
    print("[開始] エージェントに質問を送っています…", flush=True)
    progress_task = asyncio.create_task(show_progress(done, started))
    answer_printed = False

    try:
        async with asyncio.timeout(timeout):
            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=question)]),
            ):
                if event.error_code:
                    raise RuntimeError(f"{event.error_code}: {event.error_message or '詳細なし'}")
                for call in event.get_function_calls():
                    print(f"[ツール実行] {call.name}", flush=True)
                for response in event.get_function_responses():
                    print(f"[ツール完了] {response.name}。回答を待っています…", flush=True)

                if event.is_final_response() and event.content and event.content.parts:
                    answer = "".join(
                        part.text for part in event.content.parts if part.text and not part.thought
                    )
                    if answer:
                        print(f"\n{answer}\n", flush=True)
                        answer_printed = True
        if not answer_printed:
            print("[完了] 表示できる回答はありませんでした。", flush=True)
    finally:
        done.set()
        await progress_task


async def run(question: str | None, timeout: float) -> None:
    # ADK CLI と同じ優先順位でエージェント用の .env を読み込む。
    load_dotenv_for_agent(APP_NAME, str(Path(__file__).resolve().parent.parent))
    async with InMemoryRunner(agent=root_agent, app_name=APP_NAME) as runner:
        session = await runner.session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID
        )
        if question:
            await ask(runner, session.id, question, timeout)
            return

        print("質問を入力してください。終了するには exit または Ctrl+D。", flush=True)
        while True:
            try:
                question = input("\nあなた: ").strip()
            except EOFError:
                print()
                return
            if question.lower() in {"exit", "quit"}:
                return
            if question:
                await ask(runner, session.id, question, timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description="進行状況を表示する学習支援エージェント")
    parser.add_argument("question", nargs="?", help="1回だけ質問して終了する")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="1回の質問の待機上限（秒、既定: 60）")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout は 0 より大きい秒数を指定してください")
    try:
        asyncio.run(run(args.question, args.timeout))
    except KeyboardInterrupt:
        print("\n中断しました。", file=sys.stderr)
        return 130
    except TimeoutError:
        print(f"\n[タイムアウト] {args.timeout:g}秒以内に回答を受け取れませんでした。", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[エラー] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import os
import time

from google import genai
from google.genai import errors
from openai import OpenAI, OpenAIError

from app.config import MODEL, PROVIDER, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
LOCAL_MODELS = {
    "qwen": os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    "llm-jp-4": os.getenv("LLM_JP_MODEL", "llm-jp/llm-jp-4-8b-instruct"),
}


def ask_local(question: str, model_name: str) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise SystemExit(
            "ローカルモデルには torch と transformers が必要です。"
            " `uv sync` を実行してください。"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_options = {"device_map": "auto"}
    if torch.cuda.is_available():
        load_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_options["torch_dtype"] = torch.float16
    else:
        load_options["torch_dtype"] = "auto"
    if torch.cuda.is_available():
        load_options["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_options)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    elapsed_ms = (time.perf_counter() - started) * 1000
    input_tokens = inputs["input_ids"].shape[-1]
    output_tokens = generated.shape[-1] - input_tokens
    print(
        "[usage] "
        f"input_tokens={input_tokens} "
        f"output_tokens={output_tokens} "
        f"elapsed_ms={elapsed_ms:.1f}"
    )
    return tokenizer.decode(generated[0, input_tokens:], skip_special_tokens=True)


def ask(question: str) -> str:
    if MODEL in LOCAL_MODELS:
        return ask_local(question, LOCAL_MODELS[MODEL])

    if PROVIDER == "openai" or MODEL == "openai":
        client = OpenAI(api_key=get_api_key("openai"))
        model = os.getenv("OPENAI_MODEL", "gpt-6-luna") if MODEL == "openai" else MODEL
        started = time.perf_counter()
        response = client.responses.create(model=model, input=question)
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            "[usage] "
            f"input_tokens={response.usage.input_tokens} "
            f"output_tokens={response.usage.output_tokens} "
            f"elapsed_ms={elapsed_ms:.1f}"
        )
        return response.output_text

    client = genai.Client(api_key=get_api_key("google"))
    started = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=question)
    elapsed_ms = (time.perf_counter() - started) * 1000

    usage = response.usage_metadata
    print(
        "[usage] "
        f"input_tokens={usage.prompt_token_count} "
        f"output_tokens={usage.candidates_token_count} "
        f"elapsed_ms={elapsed_ms:.1f}"
    )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    except OpenAIError as e:
        status = getattr(e, "status_code", None) or 0
        print(api_error_message(status, str(e)[:200]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

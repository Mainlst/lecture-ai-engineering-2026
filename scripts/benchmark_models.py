"""複数モデルの応答時間とトークン数を比較する。

例:
    uv run python scripts/benchmark_models.py
    OPENAI_MODEL=gpt-4.1-mini uv run python scripts/benchmark_models.py

ローカルモデルは transformers から直接ロードする。モデル名は環境変数で変更できる。
"""

import argparse
import os
import time

from openai import OpenAI

DEFAULT_QUESTIONS = [
    "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
    "LoRAとは何ですか？",
]
DEFAULT_LOCAL_MODELS = {
    "qwen": os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
    "llm-jp-4": os.getenv("LLM_JP_MODEL", "llm-jp/llm-jp-4-8b-instruct"),
}


def run_openai_once(client: OpenAI, model: str, question: str) -> dict[str, object]:
    started = time.perf_counter()
    try:
        response = client.responses.create(
            model=model,
            input=question,
            temperature=0,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        usage = response.usage
        return {
            "status": "ok",
            "input_tokens": usage.input_tokens if usage else "",
            "output_tokens": usage.output_tokens if usage else "",
            "elapsed_ms": elapsed_ms,
            "answer": response.output_text.replace("\n", " "),
            "error": "",
        }
    except Exception as error:
        return {
            "status": "error",
            "input_tokens": "",
            "output_tokens": "",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "answer": "",
            "error": str(error)[:200],
        }


def load_local_model(model_name: str):
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
    return tokenizer, model


def run_local_once(tokenizer, model, question: str) -> dict[str, object]:
    import torch

    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    input_tokens = inputs["input_ids"].shape[-1]
    output_tokens = generated.shape[-1] - input_tokens
    answer = tokenizer.decode(generated[0, input_tokens:], skip_special_tokens=True)
    return {
        "status": "ok",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "elapsed_ms": elapsed_ms,
        "answer": answer.replace("\n", " "),
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="モデルごとの応答時間・トークン数を比較する")
    parser.add_argument("--skip-openai", action="store_true", help="OpenAIを呼び出さずローカルモデルだけ測定する")
    args = parser.parse_args()

    questions = os.getenv("BENCHMARK_QUESTIONS")
    if questions:
        question_list = [question.strip() for question in questions.split("|") if question.strip()]
    else:
        question_list = DEFAULT_QUESTIONS

    openai_model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    if not args.skip_openai:
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        for question in question_list:
            print(f"[openai] {question}")
            result = run_openai_once(openai_client, openai_model, question)
            print(
                f"  {result['status']}: {result['elapsed_ms']} ms, "
                f"input={result['input_tokens']}, output={result['output_tokens']}"
            )

    selected_local_models = os.getenv("BENCHMARK_LOCAL_MODELS")
    local_models = {
        name: model
        for name, model in DEFAULT_LOCAL_MODELS.items()
        if not selected_local_models or name in selected_local_models.split(",")
    }
    for name, model_name in local_models.items():
        print(f"[{name}] モデルをロード: {model_name}")
        tokenizer, model = load_local_model(model_name)
        for question in question_list:
            print(f"[{name}] {question}")
            try:
                result = run_local_once(tokenizer, model, question)
            except Exception as error:
                result = {
                    "status": "error",
                    "input_tokens": "",
                    "output_tokens": "",
                    "elapsed_ms": "",
                    "answer": "",
                    "error": str(error)[:200],
                }
            print(
                f"  {result['status']}: {result['elapsed_ms']} ms, "
                f"input={result['input_tokens']}, output={result['output_tokens']}"
            )


if __name__ == "__main__":
    main()

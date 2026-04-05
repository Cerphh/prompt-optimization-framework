"""
Direct baseline test for llama3 using only a raw user prompt.

This script does not use the framework pipeline, prompt generator,
or technique selection logic.
"""

import argparse
import json
import sys
import time
from typing import Any, Dict

import requests


def build_payload(model: str, prompt: str, num_predict: int, num_ctx: int, temperature: float) -> Dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }


def run_prompt(
    base_url: str,
    model: str,
    prompt: str,
    num_predict: int,
    num_ctx: int,
    temperature: float,
    timeout_seconds: int,
) -> Dict[str, Any]:
    payload = build_payload(
        model=model,
        prompt=prompt,
        num_predict=num_predict,
        num_ctx=num_ctx,
        temperature=temperature,
    )

    start_time = time.time()
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json=payload,
        timeout=(5, timeout_seconds),
    )
    elapsed = time.time() - start_time
    response.raise_for_status()

    data = response.json()
    return {
        "response": data.get("response", ""),
        "done_reason": data.get("done_reason", "stop"),
        "prompt_tokens": int(data.get("prompt_eval_count", 0) or 0),
        "completion_tokens": int(data.get("eval_count", 0) or 0),
        "total_tokens": int(data.get("prompt_eval_count", 0) or 0) + int(data.get("eval_count", 0) or 0),
        "elapsed_time": elapsed,
        "load_time": float(data.get("load_duration", 0) or 0) / 1e9,
        "prompt_eval_time": float(data.get("prompt_eval_duration", 0) or 0) / 1e9,
        "eval_time": float(data.get("eval_duration", 0) or 0) / 1e9,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a direct llama3 baseline prompt without framework techniques."
    )
    parser.add_argument("--prompt", type=str, default="", help="Raw prompt to send to llama3.")
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model name.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:11434", help="Ollama base URL.")
    parser.add_argument("--num-predict", type=int, default=2048, help="Max generated tokens.")
    parser.add_argument("--num-ctx", type=int, default=8192, help="Context window.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--timeout", type=int, default=600, help="Read timeout in seconds.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full result as JSON instead of formatted text.",
    )
    return parser.parse_args()


def read_prompt_from_stdin() -> str:
    print("Enter your prompt. Finish with Ctrl+Z then Enter (Windows), or Ctrl+D on Unix.")
    return sys.stdin.read().strip()


def main() -> int:
    args = parse_args()
    prompt = args.prompt.strip() if args.prompt else read_prompt_from_stdin()

    if not prompt:
        print("Error: prompt is empty.")
        return 1

    try:
        result = run_prompt(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            num_predict=max(1, args.num_predict),
            num_ctx=max(128, args.num_ctx),
            temperature=max(0.0, args.temperature),
            timeout_seconds=max(5, args.timeout),
        )
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.text.strip() if exc.response is not None else ""
        except Exception:
            detail = ""
        print(f"HTTP error while calling Ollama: {exc}")
        if detail:
            print(detail)
        return 1
    except requests.RequestException as exc:
        print(f"Connection error while calling Ollama: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("=" * 80)
    print("BASELINE RESPONSE")
    print("=" * 80)
    print(result["response"])
    print("\n" + "=" * 80)
    print("METRICS")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Done reason: {result['done_reason']}")
    print(f"Elapsed time: {result['elapsed_time']:.3f}s")
    print(f"Prompt tokens: {result['prompt_tokens']}")
    print(f"Completion tokens: {result['completion_tokens']}")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Load time: {result['load_time']:.3f}s")
    print(f"Prompt eval time: {result['prompt_eval_time']:.3f}s")
    print(f"Eval time: {result['eval_time']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

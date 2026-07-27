import argparse
import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class BenchmarkResult:
    use_cache: bool
    run_number: int
    prompt_tokens: int
    generated_tokens: int
    generation_time_seconds: float
    tokens_per_second: float
    memory_before_gb: float
    memory_after_gb: float
    memory_increase_gb: float


def select_device() -> torch.device:
    """Select the best available inference device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    """Wait for pending device operations to complete."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def clear_memory(device: torch.device) -> None:
    """Clear Python and accelerator caches."""
    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def get_process_memory_gb() -> float:
    """Return memory used by the current process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def load_model(
    model_name: str,
    device: torch.device,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load the tokenizer and model."""
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dtype = torch.float32 if device.type == "cpu" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model


def format_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
) -> str:
    """Apply the model's chat template when one is available."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI systems research assistant. "
                "Provide a technically accurate answer."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return user_prompt


@torch.inference_mode()
def run_single_benchmark(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    use_cache: bool,
    run_number: int,
) -> BenchmarkResult:
    """Run one generation benchmark."""
    clear_memory(device)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    prompt_tokens = input_ids.shape[1]
    memory_before = get_process_memory_gb()

    synchronize_device(device)
    start_time = time.perf_counter()

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=use_cache,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    synchronize_device(device)
    end_time = time.perf_counter()

    memory_after = get_process_memory_gb()

    generated_tokens = output_ids.shape[1] - prompt_tokens
    generation_time = end_time - start_time

    tokens_per_second = (
        generated_tokens / generation_time
        if generation_time > 0
        else 0.0
    )

    return BenchmarkResult(
        use_cache=use_cache,
        run_number=run_number,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        generation_time_seconds=generation_time,
        tokens_per_second=tokens_per_second,
        memory_before_gb=memory_before,
        memory_after_gb=memory_after,
        memory_increase_gb=memory_after - memory_before,
    )


def run_warmup(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    device: torch.device,
) -> None:
    """Run a short generation before measuring performance."""
    print("\nRunning warm-up generation...")

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.inference_mode():
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    synchronize_device(device)
    clear_memory(device)

    print("Warm-up complete.")


def print_individual_result(result: BenchmarkResult) -> None:
    """Print measurements from one benchmark run."""
    cache_label = "Enabled" if result.use_cache else "Disabled"

    print(
        f"Cache: {cache_label:<8} | "
        f"Run: {result.run_number} | "
        f"Tokens: {result.generated_tokens:<4} | "
        f"Time: {result.generation_time_seconds:>8.3f}s | "
        f"Speed: {result.tokens_per_second:>8.2f} tokens/s | "
        f"Memory increase: {result.memory_increase_gb:>6.3f} GB"
    )


def summarize_results(
    results: list[BenchmarkResult],
    use_cache: bool,
) -> dict[str, float]:
    """Calculate average benchmark statistics."""
    selected = [
        result
        for result in results
        if result.use_cache == use_cache
    ]

    return {
        "average_time": statistics.mean(
            result.generation_time_seconds for result in selected
        ),
        "average_speed": statistics.mean(
            result.tokens_per_second for result in selected
        ),
        "average_memory_increase": statistics.mean(
            result.memory_increase_gb for result in selected
        ),
    }


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print average results and cache speedup."""
    cached = summarize_results(results, use_cache=True)
    uncached = summarize_results(results, use_cache=False)

    speedup = (
        cached["average_speed"] / uncached["average_speed"]
        if uncached["average_speed"] > 0
        else 0.0
    )

    time_reduction_percent = (
        (
            uncached["average_time"] - cached["average_time"]
        )
        / uncached["average_time"]
        * 100
        if uncached["average_time"] > 0
        else 0.0
    )

    print("\n" + "=" * 80)
    print("AVERAGE RESULTS")
    print("=" * 80)

    print("\nKV cache enabled")
    print(f"Average time            : {cached['average_time']:.3f} seconds")
    print(f"Average generation speed: {cached['average_speed']:.2f} tokens/second")
    print(
        f"Average memory increase : "
        f"{cached['average_memory_increase']:.3f} GB"
    )

    print("\nKV cache disabled")
    print(f"Average time            : {uncached['average_time']:.3f} seconds")
    print(f"Average generation speed: {uncached['average_speed']:.2f} tokens/second")
    print(
        f"Average memory increase : "
        f"{uncached['average_memory_increase']:.3f} GB"
    )

    print("\nComparison")
    print(f"KV-cache speedup        : {speedup:.2f}x")
    print(f"Generation-time reduction: {time_reduction_percent:.2f}%")

    print("\nInterpretation:")
    print(
        "With the KV cache enabled, the model reuses attention keys and "
        "values calculated for previous tokens."
    )
    print(
        "Without the cache, the model repeatedly recomputes attention "
        "states for the full sequence during every decoding step."
    )


def parse_arguments() -> argparse.Namespace:
    """Read command-line benchmark options."""
    parser = argparse.ArgumentParser(
        description="Compare transformer generation with and without KV cache."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model name.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Explain how continuous batching and paged KV-cache management "
            "improve large language model inference throughput."
        ),
        help="Prompt used for benchmarking.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of generated tokens per run.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measured runs for each cache configuration.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    device = select_device()

    print("=" * 80)
    print("KV-CACHE BASELINE BENCHMARK")
    print("=" * 80)
    print(f"Model          : {args.model}")
    print(f"Device         : {device}")
    print(f"Runs           : {args.runs}")
    print(f"New tokens     : {args.max_new_tokens}")

    tokenizer, model = load_model(
        model_name=args.model,
        device=device,
    )

    prompt = format_prompt(
        tokenizer=tokenizer,
        user_prompt=args.prompt,
    )

    run_warmup(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        device=device,
    )

    results: list[BenchmarkResult] = []

    print("\n" + "=" * 80)
    print("BENCHMARK RUNS")
    print("=" * 80)

    for use_cache in (True, False):
        for run_number in range(1, args.runs + 1):
            result = run_single_benchmark(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                use_cache=use_cache,
                run_number=run_number,
            )

            results.append(result)
            print_individual_result(result)

    print_summary(results)


if __name__ == "__main__":
    main()
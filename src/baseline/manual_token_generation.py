import argparse
import gc
import statistics
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class GenerationMetrics:
    mode: str
    prompt_tokens: int
    generated_tokens: int
    prefill_time_seconds: float
    decode_time_seconds: float
    total_time_seconds: float
    first_token_latency_seconds: float
    average_decode_latency_ms: float
    decode_tokens_per_second: float
    total_tokens_per_second: float
    per_token_latencies_ms: list[float]
    generated_text: str


def select_device() -> torch.device:
    """Select the best available PyTorch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator operations to complete."""
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


def load_model(
    model_name: str,
    device: torch.device,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load the tokenizer and model."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dtype = torch.float32 if device.type == "cpu" else torch.float16

    print(f"Loading model with dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model = model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


def format_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
) -> str:
    """Apply the model's chat template when available."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI-systems research assistant. "
                "Answer technically and clearly."
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


def choose_next_token(logits: torch.Tensor) -> torch.Tensor:
    """
    Select the next token using greedy decoding.

    logits shape:
        [batch_size, sequence_length, vocabulary_size]
    """
    next_token_logits = logits[:, -1, :]
    return torch.argmax(
        next_token_logits,
        dim=-1,
        keepdim=True,
    )


def calculate_metrics(
    *,
    mode: str,
    prompt_tokens: int,
    generated_token_ids: list[int],
    prefill_time: float,
    decode_time: float,
    per_token_latencies: list[float],
    tokenizer: AutoTokenizer,
) -> GenerationMetrics:
    """Create a metrics object from generation measurements."""
    generated_tokens = len(generated_token_ids)
    total_time = prefill_time + decode_time

    average_decode_latency_ms = (
        statistics.mean(per_token_latencies) * 1000
        if per_token_latencies
        else 0.0
    )

    decode_tokens_per_second = (
        generated_tokens / decode_time
        if decode_time > 0
        else 0.0
    )

    total_tokens_per_second = (
        generated_tokens / total_time
        if total_time > 0
        else 0.0
    )

    generated_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    )

    return GenerationMetrics(
        mode=mode,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        prefill_time_seconds=prefill_time,
        decode_time_seconds=decode_time,
        total_time_seconds=total_time,
        first_token_latency_seconds=(
            prefill_time + per_token_latencies[0]
            if per_token_latencies
            else prefill_time
        ),
        average_decode_latency_ms=average_decode_latency_ms,
        decode_tokens_per_second=decode_tokens_per_second,
        total_tokens_per_second=total_tokens_per_second,
        per_token_latencies_ms=[
            latency * 1000 for latency in per_token_latencies
        ],
        generated_text=generated_text,
    )


@torch.inference_mode()
def generate_with_cache(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
) -> GenerationMetrics:
    """
    Generate tokens using a KV cache.

    Prefill:
        Process the entire prompt once.

    Decode:
        Pass only the newly generated token while reusing past_key_values.
    """
    clear_memory(device)

    prompt_tokens = input_ids.shape[1]

    synchronize_device(device)
    prefill_start = time.perf_counter()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    synchronize_device(device)
    prefill_end = time.perf_counter()

    prefill_time = prefill_end - prefill_start
    past_key_values = outputs.past_key_values

    next_token = choose_next_token(outputs.logits)
    generated_token_ids: list[int] = []
    per_token_latencies: list[float] = []

    decode_start_total = time.perf_counter()

    for token_index in range(max_new_tokens):
        if token_index == 0:
            token_start = time.perf_counter()

            generated_token_ids.append(next_token.item())

            synchronize_device(device)
            token_end = time.perf_counter()
            per_token_latencies.append(token_end - token_start)

        else:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )

            token_start = time.perf_counter()

            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            next_token = choose_next_token(outputs.logits)
            past_key_values = outputs.past_key_values

            synchronize_device(device)
            token_end = time.perf_counter()

            generated_token_ids.append(next_token.item())
            per_token_latencies.append(token_end - token_start)

        if next_token.item() == tokenizer.eos_token_id:
            break

    decode_end_total = time.perf_counter()
    decode_time = decode_end_total - decode_start_total

    return calculate_metrics(
        mode="KV cache enabled",
        prompt_tokens=prompt_tokens,
        generated_token_ids=generated_token_ids,
        prefill_time=prefill_time,
        decode_time=decode_time,
        per_token_latencies=per_token_latencies,
        tokenizer=tokenizer,
    )


@torch.inference_mode()
def generate_without_cache(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
) -> GenerationMetrics:
    """
    Generate tokens without a KV cache.

    The full prompt plus every previously generated token is passed through
    the model again at every decoding step.
    """
    clear_memory(device)

    prompt_tokens = input_ids.shape[1]
    current_input_ids = input_ids.clone()
    current_attention_mask = attention_mask.clone()

    generated_token_ids: list[int] = []
    per_token_latencies: list[float] = []

    prefill_time = 0.0
    total_start = time.perf_counter()

    for token_index in range(max_new_tokens):
        synchronize_device(device)
        token_start = time.perf_counter()

        outputs = model(
            input_ids=current_input_ids,
            attention_mask=current_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        next_token = choose_next_token(outputs.logits)

        synchronize_device(device)
        token_end = time.perf_counter()

        token_latency = token_end - token_start
        per_token_latencies.append(token_latency)

        if token_index == 0:
            prefill_time = token_latency

        generated_token_ids.append(next_token.item())

        current_input_ids = torch.cat(
            [current_input_ids, next_token],
            dim=1,
        )

        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones(
                    (current_attention_mask.shape[0], 1),
                    dtype=current_attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

        if next_token.item() == tokenizer.eos_token_id:
            break

    total_end = time.perf_counter()
    total_time = total_end - total_start

    decode_time = max(total_time - prefill_time, 0.0)

    return calculate_metrics(
        mode="KV cache disabled",
        prompt_tokens=prompt_tokens,
        generated_token_ids=generated_token_ids,
        prefill_time=prefill_time,
        decode_time=decode_time,
        per_token_latencies=per_token_latencies,
        tokenizer=tokenizer,
    )


def run_warmup(
    *,
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> None:
    """Run a short forward pass before benchmarking."""
    print("Running warm-up...")

    with torch.inference_mode():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

    synchronize_device(device)
    clear_memory(device)

    print("Warm-up completed.")


def print_result(result: GenerationMetrics) -> None:
    """Print generation output and timing measurements."""
    print("\n" + "=" * 80)
    print(result.mode.upper())
    print("=" * 80)

    print("\nGenerated text:")
    print(result.generated_text.strip())

    print("\nMetrics:")
    print(f"Prompt tokens              : {result.prompt_tokens}")
    print(f"Generated tokens           : {result.generated_tokens}")
    print(
        f"Prefill time               : "
        f"{result.prefill_time_seconds:.4f} seconds"
    )
    print(
        f"Decode time                : "
        f"{result.decode_time_seconds:.4f} seconds"
    )
    print(
        f"Total generation time      : "
        f"{result.total_time_seconds:.4f} seconds"
    )
    print(
        f"Time to first token        : "
        f"{result.first_token_latency_seconds:.4f} seconds"
    )
    print(
        f"Average token latency      : "
        f"{result.average_decode_latency_ms:.2f} ms"
    )
    print(
        f"Decode throughput          : "
        f"{result.decode_tokens_per_second:.2f} tokens/second"
    )
    print(
        f"End-to-end throughput      : "
        f"{result.total_tokens_per_second:.2f} tokens/second"
    )

    if result.per_token_latencies_ms:
        preview = result.per_token_latencies_ms[:10]

        print("\nFirst token latencies:")
        print(
            ", ".join(
                f"{latency:.2f} ms"
                for latency in preview
            )
        )


def print_comparison(
    cached: GenerationMetrics,
    uncached: GenerationMetrics,
) -> None:
    """Compare cached and uncached decoding."""
    speedup = (
        cached.total_tokens_per_second
        / uncached.total_tokens_per_second
        if uncached.total_tokens_per_second > 0
        else 0.0
    )

    latency_reduction = (
        (
            uncached.average_decode_latency_ms
            - cached.average_decode_latency_ms
        )
        / uncached.average_decode_latency_ms
        * 100
        if uncached.average_decode_latency_ms > 0
        else 0.0
    )

    print("\n" + "=" * 80)
    print("CACHE COMPARISON")
    print("=" * 80)

    print(
        f"Cached throughput          : "
        f"{cached.total_tokens_per_second:.2f} tokens/second"
    )
    print(
        f"Uncached throughput        : "
        f"{uncached.total_tokens_per_second:.2f} tokens/second"
    )
    print(f"Overall speedup            : {speedup:.2f}x")
    print(
        f"Average latency reduction  : "
        f"{latency_reduction:.2f}%"
    )

    print("\nResearch interpretation:")
    print(
        "Cached decoding processes approximately one new token during each "
        "decode step."
    )
    print(
        "Uncached decoding repeatedly processes the prompt and all previously "
        "generated tokens."
    )
    print(
        "As the sequence grows, uncached decoding should become progressively "
        "slower."
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure manual autoregressive generation with and without "
            "past_key_values."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model identifier.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Explain why KV-cache memory management becomes difficult "
            "when an inference server handles many concurrent requests."
        ),
        help="Prompt sent to the model.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Maximum number of tokens to generate.",
    )

    parser.add_argument(
        "--skip-no-cache",
        action="store_true",
        help="Run only cached generation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    device = select_device()

    print("=" * 80)
    print("MANUAL AUTOREGRESSIVE GENERATION")
    print("=" * 80)
    print(f"Model              : {args.model}")
    print(f"Device             : {device}")
    print(f"Maximum new tokens : {args.max_new_tokens}")

    tokenizer, model = load_model(
        model_name=args.model,
        device=device,
    )

    prompt = format_prompt(
        tokenizer=tokenizer,
        user_prompt=args.prompt,
    )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    run_warmup(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device=device,
    )

    cached_result = generate_with_cache(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    print_result(cached_result)

    if not args.skip_no_cache:
        uncached_result = generate_without_cache(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

        print_result(uncached_result)
        print_comparison(cached_result, uncached_result)


if __name__ == "__main__":
    main()
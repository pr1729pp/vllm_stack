import argparse
import gc
import os
import time
from dataclasses import dataclass

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class InferenceResult:
    prompt_tokens: int
    generated_tokens: int
    generation_time_seconds: float
    tokens_per_second: float
    memory_before_gb: float
    memory_after_gb: float
    generated_text: str


def get_process_memory_gb() -> float:
    """Return memory currently used by this Python process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


def select_device() -> torch.device:
    """Select MPS, CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator operations to finish."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def clear_device_memory(device: torch.device) -> None:
    """Release unused accelerator memory."""
    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_model(
    model_name: str,
    device: torch.device,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load the tokenizer and causal language model."""
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")

    if device.type == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)

    elif device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model = model.to(device)

    model.eval()

    return tokenizer, model


def prepare_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
) -> str:
    """
    Convert a user prompt into the model's chat-template format.

    If the tokenizer does not provide a chat template, the original prompt
    is returned.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI research assistant. "
                "Answer clearly and concisely."
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
def generate_text(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> InferenceResult:
    """Generate text and return benchmark measurements."""
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
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    synchronize_device(device)
    end_time = time.perf_counter()

    memory_after = get_process_memory_gb()

    generated_ids = output_ids[0, prompt_tokens:]
    generated_tokens = generated_ids.shape[0]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    generation_time = end_time - start_time

    tokens_per_second = (
        generated_tokens / generation_time
        if generation_time > 0
        else 0.0
    )

    return InferenceResult(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        generation_time_seconds=generation_time,
        tokens_per_second=tokens_per_second,
        memory_before_gb=memory_before,
        memory_after_gb=memory_after,
        generated_text=generated_text,
    )


def print_result(result: InferenceResult) -> None:
    """Print generated text and performance statistics."""
    print("\n" + "=" * 70)
    print("MODEL RESPONSE")
    print("=" * 70)
    print(result.generated_text.strip())

    print("\n" + "=" * 70)
    print("INFERENCE METRICS")
    print("=" * 70)
    print(f"Prompt tokens       : {result.prompt_tokens}")
    print(f"Generated tokens    : {result.generated_tokens}")
    print(
        f"Generation time     : "
        f"{result.generation_time_seconds:.3f} seconds"
    )
    print(
        f"Generation speed    : "
        f"{result.tokens_per_second:.2f} tokens/second"
    )
    print(
        f"Process memory before: "
        f"{result.memory_before_gb:.2f} GB"
    )
    print(
        f"Process memory after : "
        f"{result.memory_after_gb:.2f} GB"
    )
    print(
        f"Memory increase      : "
        f"{result.memory_after_gb - result.memory_before_gb:.2f} GB"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline Hugging Face LLM inference."
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
            "Explain the purpose of a key-value cache in transformer "
            "inference in five sentences."
        ),
        help="Prompt sent to the model.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of output tokens.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    device = select_device()

    print("=" * 70)
    print("BASELINE LANGUAGE MODEL INFERENCE")
    print("=" * 70)
    print(f"Model              : {args.model}")
    print(f"Selected device    : {device}")
    print(f"Maximum new tokens : {args.max_new_tokens}")

    clear_device_memory(device)

    tokenizer, model = load_model(
        model_name=args.model,
        device=device,
    )

    formatted_prompt = prepare_prompt(
        tokenizer=tokenizer,
        user_prompt=args.prompt,
    )

    result = generate_text(
        tokenizer=tokenizer,
        model=model,
        prompt=formatted_prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    print_result(result)


if __name__ == "__main__":
    main()
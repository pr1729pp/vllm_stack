import argparse
import gc
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

BYTES_PER_KB = 1024
BYTES_PER_MB = 1024**2
BYTES_PER_GB = 1024**3


@dataclass
class LayerCacheInfo:
    """KV-cache information for one transformer layer."""

    layer_index: int
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_dtype: torch.dtype
    value_dtype: torch.dtype
    key_memory_bytes: int
    value_memory_bytes: int

    @property
    def total_memory_bytes(self) -> int:
        return self.key_memory_bytes + self.value_memory_bytes


def select_device() -> torch.device:
    """Select MPS, CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def synchronize_device(device: torch.device) -> None:
    """Wait for pending accelerator operations."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def clear_memory(device: torch.device) -> None:
    """Release unused Python and accelerator memory."""
    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def format_bytes(number_of_bytes: float) -> str:
    """Convert a byte count into a readable unit."""
    if number_of_bytes >= BYTES_PER_GB:
        return f"{number_of_bytes / BYTES_PER_GB:.4f} GB"

    if number_of_bytes >= BYTES_PER_MB:
        return f"{number_of_bytes / BYTES_PER_MB:.4f} MB"

    if number_of_bytes >= BYTES_PER_KB:
        return f"{number_of_bytes / BYTES_PER_KB:.4f} KB"

    return f"{number_of_bytes:.0f} bytes"


def tensor_memory_bytes(tensor: torch.Tensor) -> int:
    """Calculate memory occupied by a tensor."""
    return tensor.numel() * tensor.element_size()


def load_model(
    model_name: str,
    device: torch.device,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load the tokenizer and language model."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dtype = torch.float32 if device.type == "cpu" else torch.float16

    print(f"Loading model using {dtype}...")
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
    """Apply the model chat template when available."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI inference-systems researcher. "
                "Answer clearly and technically."
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


def get_config_value(
    config: Any,
    possible_names: list[str],
    default: Any = None,
) -> Any:
    """
    Read a configuration value while supporting different model families.

    Different model architectures sometimes use different field names for
    equivalent properties.
    """
    for name in possible_names:
        value = getattr(config, name, None)

        if value is not None:
            return value

    return default


def extract_legacy_cache(
    past_key_values: Any,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Convert the returned cache into a layer-wise tuple of key/value tensors.

    New Transformers versions may return a Cache object. Older versions may
    return a tuple directly.
    """
    if past_key_values is None:
        raise RuntimeError(
            "The model did not return past_key_values. "
            "Confirm that use_cache=True is supported."
        )

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
    else:
        legacy_cache = past_key_values

    extracted_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

    for layer_index, layer_cache in enumerate(legacy_cache):
        if not isinstance(layer_cache, (tuple, list)):
            raise TypeError(
                f"Unexpected cache format in layer {layer_index}: "
                f"{type(layer_cache)}"
            )

        if len(layer_cache) < 2:
            raise ValueError(
                f"Layer {layer_index} does not contain key and value tensors."
            )

        key_tensor = layer_cache[0]
        value_tensor = layer_cache[1]

        if not isinstance(key_tensor, torch.Tensor):
            raise TypeError(
                f"Layer {layer_index} key is not a tensor."
            )

        if not isinstance(value_tensor, torch.Tensor):
            raise TypeError(
                f"Layer {layer_index} value is not a tensor."
            )

        extracted_layers.append((key_tensor, value_tensor))

    return tuple(extracted_layers)


def inspect_layers(
    legacy_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> list[LayerCacheInfo]:
    """Collect tensor shape and memory information for every layer."""
    layer_information: list[LayerCacheInfo] = []

    for layer_index, (key_tensor, value_tensor) in enumerate(legacy_cache):
        information = LayerCacheInfo(
            layer_index=layer_index,
            key_shape=tuple(key_tensor.shape),
            value_shape=tuple(value_tensor.shape),
            key_dtype=key_tensor.dtype,
            value_dtype=value_tensor.dtype,
            key_memory_bytes=tensor_memory_bytes(key_tensor),
            value_memory_bytes=tensor_memory_bytes(value_tensor),
        )

        layer_information.append(information)

    return layer_information


def print_model_configuration(
    model: AutoModelForCausalLM,
) -> None:
    """Print attention-related model configuration."""
    config = model.config

    hidden_size = get_config_value(
        config,
        ["hidden_size", "n_embd", "d_model"],
    )

    number_of_layers = get_config_value(
        config,
        ["num_hidden_layers", "n_layer", "num_layers"],
    )

    number_of_attention_heads = get_config_value(
        config,
        ["num_attention_heads", "n_head"],
    )

    number_of_kv_heads = get_config_value(
        config,
        ["num_key_value_heads", "num_kv_heads"],
        number_of_attention_heads,
    )

    head_dimension = get_config_value(
        config,
        ["head_dim"],
    )

    if (
        head_dimension is None
        and hidden_size is not None
        and number_of_attention_heads is not None
    ):
        head_dimension = hidden_size // number_of_attention_heads

    print("\n" + "=" * 80)
    print("MODEL CONFIGURATION")
    print("=" * 80)

    print(f"Model architecture        : {config.model_type}")
    print(f"Hidden size               : {hidden_size}")
    print(f"Transformer layers        : {number_of_layers}")
    print(f"Attention heads           : {number_of_attention_heads}")
    print(f"Key/value heads           : {number_of_kv_heads}")
    print(f"Head dimension            : {head_dimension}")

    if number_of_attention_heads and number_of_kv_heads:
        query_heads_per_kv_head = (
            number_of_attention_heads / number_of_kv_heads
        )

        print(
            f"Query heads per KV head    : "
            f"{query_heads_per_kv_head:.2f}"
        )

        if number_of_attention_heads == number_of_kv_heads:
            print("Attention type             : Multi-head attention")
        elif number_of_kv_heads == 1:
            print("Attention type             : Multi-query attention")
        else:
            print("Attention type             : Grouped-query attention")


def print_layer_information(
    layer_information: list[LayerCacheInfo],
) -> None:
    """Print KV-cache tensor details for every transformer layer."""
    print("\n" + "=" * 80)
    print("KV-CACHE TENSORS BY LAYER")
    print("=" * 80)

    for layer in layer_information:
        print(f"\nLayer {layer.layer_index}")
        print(f"  Key shape       : {layer.key_shape}")
        print(f"  Value shape     : {layer.value_shape}")
        print(f"  Key dtype       : {layer.key_dtype}")
        print(f"  Value dtype     : {layer.value_dtype}")
        print(
            f"  Key memory      : "
            f"{format_bytes(layer.key_memory_bytes)}"
        )
        print(
            f"  Value memory    : "
            f"{format_bytes(layer.value_memory_bytes)}"
        )
        print(
            f"  Total layer KV  : "
            f"{format_bytes(layer.total_memory_bytes)}"
        )


def print_memory_summary(
    layer_information: list[LayerCacheInfo],
    prompt_tokens: int,
    batch_size: int,
) -> None:
    """Print actual and per-token KV-cache memory."""
    total_cache_bytes = sum(
        layer.total_memory_bytes
        for layer in layer_information
    )

    bytes_per_sequence_token = (
        total_cache_bytes / (prompt_tokens * batch_size)
        if prompt_tokens > 0 and batch_size > 0
        else 0.0
    )

    bytes_per_request = (
        total_cache_bytes / batch_size
        if batch_size > 0
        else 0.0
    )

    print("\n" + "=" * 80)
    print("ACTUAL CACHE MEMORY")
    print("=" * 80)

    print(f"Batch size                    : {batch_size}")
    print(f"Prompt tokens per request     : {prompt_tokens}")
    print(f"Number of cached layers       : {len(layer_information)}")
    print(
        f"Total KV-cache memory         : "
        f"{format_bytes(total_cache_bytes)}"
    )
    print(
        f"KV cache per request          : "
        f"{format_bytes(bytes_per_request)}"
    )
    print(
        f"KV cache per sequence token   : "
        f"{format_bytes(bytes_per_sequence_token)}"
    )

    context_lengths = [128, 512, 1024, 2048, 4096, 8192, 32768]

    print("\nEstimated KV cache for one request:")

    for context_length in context_lengths:
        estimated_memory = bytes_per_sequence_token * context_length

        print(
            f"  {context_length:>6} tokens : "
            f"{format_bytes(estimated_memory)}"
        )


def calculate_theoretical_memory(
    model: AutoModelForCausalLM,
    dtype: torch.dtype,
) -> float | None:
    """
    Calculate theoretical KV-cache bytes per token.

    Formula:

        2 × layers × KV heads × head dimension × bytes per element

    The factor 2 represents the key and value tensors.
    """
    config = model.config

    hidden_size = get_config_value(
        config,
        ["hidden_size", "n_embd", "d_model"],
    )

    number_of_layers = get_config_value(
        config,
        ["num_hidden_layers", "n_layer", "num_layers"],
    )

    number_of_attention_heads = get_config_value(
        config,
        ["num_attention_heads", "n_head"],
    )

    number_of_kv_heads = get_config_value(
        config,
        ["num_key_value_heads", "num_kv_heads"],
        number_of_attention_heads,
    )

    head_dimension = get_config_value(
        config,
        ["head_dim"],
    )

    if (
        head_dimension is None
        and hidden_size is not None
        and number_of_attention_heads is not None
    ):
        head_dimension = hidden_size // number_of_attention_heads

    required_values = [
        number_of_layers,
        number_of_kv_heads,
        head_dimension,
    ]

    if any(value is None for value in required_values):
        return None

    bytes_per_element = torch.tensor(
        [],
        dtype=dtype,
    ).element_size()

    theoretical_bytes_per_token = (
        2
        * number_of_layers
        * number_of_kv_heads
        * head_dimension
        * bytes_per_element
    )

    return float(theoretical_bytes_per_token)


def print_theoretical_comparison(
    model: AutoModelForCausalLM,
    layer_information: list[LayerCacheInfo],
    prompt_tokens: int,
    batch_size: int,
) -> None:
    """Compare formula-based memory with actual tensor memory."""
    if not layer_information:
        return

    cache_dtype = layer_information[0].key_dtype

    theoretical_bytes = calculate_theoretical_memory(
        model=model,
        dtype=cache_dtype,
    )

    actual_total_bytes = sum(
        layer.total_memory_bytes
        for layer in layer_information
    )

    actual_bytes_per_token = (
        actual_total_bytes / (prompt_tokens * batch_size)
    )

    print("\n" + "=" * 80)
    print("THEORETICAL VS ACTUAL MEMORY")
    print("=" * 80)

    if theoretical_bytes is None:
        print(
            "The theoretical estimate could not be calculated because "
            "the model configuration uses unsupported field names."
        )
        return

    difference = actual_bytes_per_token - theoretical_bytes

    print(
        f"Theoretical bytes per token : "
        f"{format_bytes(theoretical_bytes)}"
    )
    print(
        f"Actual bytes per token      : "
        f"{format_bytes(actual_bytes_per_token)}"
    )
    print(
        f"Difference                  : "
        f"{format_bytes(abs(difference))}"
    )

    print("\nFormula:")
    print(
        "2 × number of layers × number of KV heads × "
        "head dimension × bytes per element"
    )


@torch.inference_mode()
def create_kv_cache(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> Any:
    """Run prefill and return the resulting KV cache."""
    clear_memory(device)
    synchronize_device(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    synchronize_device(device)

    return outputs.past_key_values


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect transformer KV-cache tensors and memory usage."
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
            "Explain how paged memory allocation can reduce KV-cache "
            "fragmentation in an LLM inference server."
        ),
        help="Prompt used to create the cache.",
    )

    parser.add_argument(
        "--repeat-prompt",
        type=int,
        default=1,
        help=(
            "Repeat the prompt to produce a longer input and larger cache."
        ),
    )

    parser.add_argument(
        "--show-all-layers",
        action="store_true",
        help="Print information for every transformer layer.",
    )

    return parser.parse_args()


def main() -> None:
    """Load a model, create its cache, and inspect the cache."""
    args = parse_arguments()

    if args.repeat_prompt < 1:
        raise ValueError("--repeat-prompt must be at least 1.")

    device = select_device()

    print("=" * 80)
    print("KV-CACHE INSPECTION")
    print("=" * 80)
    print(f"Model             : {args.model}")
    print(f"Device            : {device}")
    print(f"Prompt repetitions: {args.repeat_prompt}")

    tokenizer, model = load_model(
        model_name=args.model,
        device=device,
    )

    repeated_prompt = " ".join(
        [args.prompt] * args.repeat_prompt
    )

    formatted_prompt = format_prompt(
        tokenizer=tokenizer,
        user_prompt=repeated_prompt,
    )

    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    batch_size = input_ids.shape[0]
    prompt_tokens = input_ids.shape[1]

    print(f"Batch size        : {batch_size}")
    print(f"Prompt tokens     : {prompt_tokens}")

    print_model_configuration(model)

    past_key_values = create_kv_cache(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device=device,
    )

    legacy_cache = extract_legacy_cache(past_key_values)
    layer_information = inspect_layers(legacy_cache)

    if args.show_all_layers:
        print_layer_information(layer_information)
    else:
        print("\n" + "=" * 80)
        print("FIRST AND LAST CACHE LAYERS")
        print("=" * 80)

        selected_layers = [layer_information[0]]

        if len(layer_information) > 1:
            selected_layers.append(layer_information[-1])

        print_layer_information(selected_layers)

        print(
            "\nUse --show-all-layers to display every transformer layer."
        )

    print_memory_summary(
        layer_information=layer_information,
        prompt_tokens=prompt_tokens,
        batch_size=batch_size,
    )

    print_theoretical_comparison(
        model=model,
        layer_information=layer_information,
        prompt_tokens=prompt_tokens,
        batch_size=batch_size,
    )

    print("\n" + "=" * 80)
    print("RESEARCH INTERPRETATION")
    print("=" * 80)
    print(
        "KV-cache memory grows approximately linearly with the number of "
        "cached tokens, active requests, transformer layers, KV heads, "
        "and head dimension."
    )
    print(
        "vLLM manages this growing memory using fixed-size blocks rather "
        "than requiring one large contiguous cache allocation per request."
    )


if __name__ == "__main__":
    main()
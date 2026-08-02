import argparse
import random
from dataclasses import dataclass

from request import Request


@dataclass
class WorkloadConfig:
    """Configuration for generating simulated inference requests."""

    number_of_requests: int = 20
    arrival_pattern: str = "random"
    minimum_prompt_tokens: int = 16
    maximum_prompt_tokens: int = 256
    minimum_output_tokens: int = 4
    maximum_output_tokens: int = 64
    maximum_arrival_step: int = 20
    seed: int = 42


class RequestGenerator:
    """Generate synthetic LLM inference requests."""

    SUPPORTED_PATTERNS = {
        "all-at-once",
        "random",
        "steady",
        "bursty",
    }

    def __init__(self, config: WorkloadConfig) -> None:
        self.config = config
        self._validate_config()
        self.random = random.Random(config.seed)

    def _validate_config(self) -> None:
        """Validate workload configuration values."""
        if self.config.number_of_requests <= 0:
            raise ValueError(
                "number_of_requests must be greater than zero."
            )

        if self.config.arrival_pattern not in self.SUPPORTED_PATTERNS:
            supported = ", ".join(sorted(self.SUPPORTED_PATTERNS))

            raise ValueError(
                f"Unsupported arrival pattern: "
                f"{self.config.arrival_pattern}. "
                f"Supported patterns: {supported}."
            )

        if self.config.minimum_prompt_tokens <= 0:
            raise ValueError(
                "minimum_prompt_tokens must be greater than zero."
            )

        if (
            self.config.maximum_prompt_tokens
            < self.config.minimum_prompt_tokens
        ):
            raise ValueError(
                "maximum_prompt_tokens must be greater than or equal "
                "to minimum_prompt_tokens."
            )

        if self.config.minimum_output_tokens <= 0:
            raise ValueError(
                "minimum_output_tokens must be greater than zero."
            )

        if (
            self.config.maximum_output_tokens
            < self.config.minimum_output_tokens
        ):
            raise ValueError(
                "maximum_output_tokens must be greater than or equal "
                "to minimum_output_tokens."
            )

        if self.config.maximum_arrival_step < 0:
            raise ValueError(
                "maximum_arrival_step cannot be negative."
            )

    def _generate_arrival_steps(self) -> list[int]:
        """Generate request arrival times using the selected pattern."""
        pattern = self.config.arrival_pattern
        request_count = self.config.number_of_requests

        if pattern == "all-at-once":
            return [0] * request_count

        if pattern == "steady":
            return list(range(request_count))

        if pattern == "random":
            arrival_steps = [
                self.random.randint(
                    0,
                    self.config.maximum_arrival_step,
                )
                for _ in range(request_count)
            ]

            return sorted(arrival_steps)

        if pattern == "bursty":
            return self._generate_bursty_arrivals()

        raise RuntimeError(
            f"Arrival pattern was not handled: {pattern}"
        )

    def _generate_bursty_arrivals(self) -> list[int]:
        """
        Generate groups of requests that arrive close together.

        Example:

            step 0:  requests 1, 2, 3
            step 5:  requests 4, 5
            step 12: requests 6, 7, 8, 9
        """
        arrival_steps: list[int] = []
        current_step = 0

        while len(arrival_steps) < self.config.number_of_requests:
            burst_size = self.random.randint(2, 5)

            remaining = (
                self.config.number_of_requests
                - len(arrival_steps)
            )

            burst_size = min(burst_size, remaining)

            arrival_steps.extend(
                [current_step] * burst_size
            )

            gap = self.random.randint(2, 7)
            current_step += gap

        return arrival_steps

    def generate(self) -> list[Request]:
        """Generate the complete workload."""
        arrival_steps = self._generate_arrival_steps()

        requests: list[Request] = []

        for index, arrival_step in enumerate(
            arrival_steps,
            start=1,
        ):
            prompt_tokens = self.random.randint(
                self.config.minimum_prompt_tokens,
                self.config.maximum_prompt_tokens,
            )

            output_tokens = self.random.randint(
                self.config.minimum_output_tokens,
                self.config.maximum_output_tokens,
            )

            request = Request(
                request_id=f"request-{index:03d}",
                arrival_step=arrival_step,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )

            requests.append(request)

        return requests


def print_requests(requests: list[Request]) -> None:
    """Print the generated workload as a table."""
    print("\n" + "=" * 72)
    print("GENERATED INFERENCE WORKLOAD")
    print("=" * 72)

    print(
        f"{'Request ID':<14}"
        f"{'Arrival':>10}"
        f"{'Prompt':>12}"
        f"{'Output':>12}"
        f"{'Total':>12}"
    )

    print("-" * 72)

    for request in requests:
        total_tokens = (
            request.prompt_tokens
            + request.output_tokens
        )

        print(
            f"{request.request_id:<14}"
            f"{request.arrival_step:>10}"
            f"{request.prompt_tokens:>12}"
            f"{request.output_tokens:>12}"
            f"{total_tokens:>12}"
        )

    print("-" * 72)

    total_prompt_tokens = sum(
        request.prompt_tokens
        for request in requests
    )

    total_output_tokens = sum(
        request.output_tokens
        for request in requests
    )

    print(f"Number of requests : {len(requests)}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(
        f"Total workload     : "
        f"{total_prompt_tokens + total_output_tokens} tokens"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic workloads for the continuous "
            "batching simulator."
        )
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=20,
        help="Number of requests to generate.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="random",
        choices=sorted(RequestGenerator.SUPPORTED_PATTERNS),
        help="Request arrival pattern.",
    )

    parser.add_argument(
        "--min-prompt",
        type=int,
        default=16,
        help="Minimum prompt length.",
    )

    parser.add_argument(
        "--max-prompt",
        type=int,
        default=256,
        help="Maximum prompt length.",
    )

    parser.add_argument(
        "--min-output",
        type=int,
        default=4,
        help="Minimum generated output length.",
    )

    parser.add_argument(
        "--max-output",
        type=int,
        default=64,
        help="Maximum generated output length.",
    )

    parser.add_argument(
        "--max-arrival-step",
        type=int,
        default=20,
        help="Maximum arrival step for random traffic.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible workloads.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config = WorkloadConfig(
        number_of_requests=args.requests,
        arrival_pattern=args.pattern,
        minimum_prompt_tokens=args.min_prompt,
        maximum_prompt_tokens=args.max_prompt,
        minimum_output_tokens=args.min_output,
        maximum_output_tokens=args.max_output,
        maximum_arrival_step=args.max_arrival_step,
        seed=args.seed,
    )

    generator = RequestGenerator(config)
    requests = generator.generate()

    print(f"Arrival pattern: {config.arrival_pattern}")
    print(f"Random seed    : {config.seed}")

    print_requests(requests)


if __name__ == "__main__":
    main()
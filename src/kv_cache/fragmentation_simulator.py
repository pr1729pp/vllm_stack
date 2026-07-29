import argparse
import math
import random
from dataclasses import dataclass


@dataclass
class AllocationResult:
    request_id: str
    requested_tokens: int
    allocated_tokens: int
    success: bool
    reason: str = ""


class ContiguousAllocator:
    """
    Simulates contiguous KV-cache allocation.

    Each request needs one uninterrupted memory region.
    """

    def __init__(self, total_tokens: int) -> None:
        self.total_tokens = total_tokens
        self.free_segments: list[tuple[int, int]] = [(0, total_tokens)]
        self.allocations: dict[str, tuple[int, int, int]] = {}

    def allocate(
        self,
        request_id: str,
        token_count: int,
    ) -> AllocationResult:
        if request_id in self.allocations:
            raise ValueError(f"{request_id} already exists.")

        for index, (start, length) in enumerate(self.free_segments):
            if length >= token_count:
                self.allocations[request_id] = (
                    start,
                    token_count,
                    token_count,
                )

                remaining = length - token_count

                if remaining == 0:
                    self.free_segments.pop(index)
                else:
                    self.free_segments[index] = (
                        start + token_count,
                        remaining,
                    )

                return AllocationResult(
                    request_id=request_id,
                    requested_tokens=token_count,
                    allocated_tokens=token_count,
                    success=True,
                )

        total_free = self.total_free_tokens()
        largest_segment = self.largest_free_segment()

        return AllocationResult(
            request_id=request_id,
            requested_tokens=token_count,
            allocated_tokens=0,
            success=False,
            reason=(
                f"Contiguous region unavailable. "
                f"Total free={total_free}, "
                f"largest free segment={largest_segment}."
            ),
        )

    def free(self, request_id: str) -> None:
        if request_id not in self.allocations:
            raise KeyError(f"{request_id} does not exist.")

        start, allocated_tokens, _ = self.allocations.pop(request_id)
        self.free_segments.append((start, allocated_tokens))
        self._merge_free_segments()

    def _merge_free_segments(self) -> None:
        self.free_segments.sort()

        merged: list[tuple[int, int]] = []

        for start, length in self.free_segments:
            if not merged:
                merged.append((start, length))
                continue

            previous_start, previous_length = merged[-1]
            previous_end = previous_start + previous_length

            if previous_end == start:
                merged[-1] = (
                    previous_start,
                    previous_length + length,
                )
            else:
                merged.append((start, length))

        self.free_segments = merged

    def total_free_tokens(self) -> int:
        return sum(length for _, length in self.free_segments)

    def largest_free_segment(self) -> int:
        if not self.free_segments:
            return 0

        return max(length for _, length in self.free_segments)

    def external_fragmentation_tokens(self) -> int:
        return self.total_free_tokens() - self.largest_free_segment()

    def used_tokens(self) -> int:
        return self.total_tokens - self.total_free_tokens()

    def print_state(self) -> None:
        print("\nCONTIGUOUS ALLOCATOR")
        print("-" * 72)
        print(f"Total capacity          : {self.total_tokens}")
        print(f"Used tokens             : {self.used_tokens()}")
        print(f"Free tokens             : {self.total_free_tokens()}")
        print(f"Largest free segment    : {self.largest_free_segment()}")
        print(
            f"External fragmentation  : "
            f"{self.external_fragmentation_tokens()}"
        )
        print(f"Free segments           : {self.free_segments}")


class PagedAllocator:
    """
    Simulates fixed-size paged KV-cache allocation.
    """

    def __init__(
        self,
        total_tokens: int,
        block_size: int,
    ) -> None:
        if total_tokens % block_size != 0:
            raise ValueError(
                "total_tokens must be divisible by block_size."
            )

        self.total_tokens = total_tokens
        self.block_size = block_size
        self.total_blocks = total_tokens // block_size
        self.free_blocks: list[int] = list(range(self.total_blocks))
        self.allocations: dict[str, tuple[list[int], int]] = {}

    def allocate(
        self,
        request_id: str,
        token_count: int,
    ) -> AllocationResult:
        if request_id in self.allocations:
            raise ValueError(f"{request_id} already exists.")

        required_blocks = math.ceil(token_count / self.block_size)

        if required_blocks > len(self.free_blocks):
            return AllocationResult(
                request_id=request_id,
                requested_tokens=token_count,
                allocated_tokens=0,
                success=False,
                reason=(
                    f"Not enough blocks. Required={required_blocks}, "
                    f"available={len(self.free_blocks)}."
                ),
            )

        assigned_blocks = [
            self.free_blocks.pop(0)
            for _ in range(required_blocks)
        ]

        self.allocations[request_id] = (
            assigned_blocks,
            token_count,
        )

        return AllocationResult(
            request_id=request_id,
            requested_tokens=token_count,
            allocated_tokens=required_blocks * self.block_size,
            success=True,
        )

    def free(self, request_id: str) -> None:
        if request_id not in self.allocations:
            raise KeyError(f"{request_id} does not exist.")

        blocks, _ = self.allocations.pop(request_id)
        self.free_blocks.extend(blocks)
        self.free_blocks.sort()

    def used_blocks(self) -> int:
        return self.total_blocks - len(self.free_blocks)

    def allocated_capacity(self) -> int:
        return self.used_blocks() * self.block_size

    def actual_tokens(self) -> int:
        return sum(
            token_count
            for _, token_count in self.allocations.values()
        )

    def internal_fragmentation_tokens(self) -> int:
        return self.allocated_capacity() - self.actual_tokens()

    def free_tokens(self) -> int:
        return len(self.free_blocks) * self.block_size

    def print_state(self) -> None:
        print("\nPAGED ALLOCATOR")
        print("-" * 72)
        print(f"Total capacity          : {self.total_tokens}")
        print(f"Block size              : {self.block_size}")
        print(f"Used blocks             : {self.used_blocks()}")
        print(f"Free blocks             : {len(self.free_blocks)}")
        print(f"Allocated capacity      : {self.allocated_capacity()}")
        print(f"Actual cached tokens    : {self.actual_tokens()}")
        print(
            f"Internal fragmentation  : "
            f"{self.internal_fragmentation_tokens()}"
        )
        print(f"Free token capacity     : {self.free_tokens()}")
        print(f"Free block IDs          : {self.free_blocks}")


def print_allocation_result(
    allocator_name: str,
    result: AllocationResult,
) -> None:
    status = "SUCCESS" if result.success else "FAILED"

    print(
        f"{allocator_name:<12} | "
        f"{result.request_id:<10} | "
        f"request={result.requested_tokens:<4} | "
        f"allocated={result.allocated_tokens:<4} | "
        f"{status}"
    )

    if result.reason:
        print(f"  Reason: {result.reason}")


def run_deterministic_demo(
    total_tokens: int,
    block_size: int,
) -> None:
    contiguous = ContiguousAllocator(total_tokens)
    paged = PagedAllocator(total_tokens, block_size)

    print("=" * 72)
    print("FRAGMENTATION DEMONSTRATION")
    print("=" * 72)

    requests = {
        "request-A": 24,
        "request-B": 40,
        "request-C": 24,
        "request-D": 24,
    }

    print("\nInitial allocations:")

    for request_id, token_count in requests.items():
        print_allocation_result(
            "Contiguous",
            contiguous.allocate(request_id, token_count),
        )
        print_allocation_result(
            "Paged",
            paged.allocate(request_id, token_count),
        )

    contiguous.print_state()
    paged.print_state()

    print("\nFreeing request-B and request-D...")

    for request_id in ("request-B", "request-D"):
        contiguous.free(request_id)
        paged.free(request_id)

    contiguous.print_state()
    paged.print_state()

    print("\nTrying to allocate request-E with 56 tokens...")

    contiguous_result = contiguous.allocate(
        "request-E",
        56,
    )

    paged_result = paged.allocate(
        "request-E",
        56,
    )

    print_allocation_result("Contiguous", contiguous_result)
    print_allocation_result("Paged", paged_result)

    contiguous.print_state()
    paged.print_state()


def run_random_simulation(
    total_tokens: int,
    block_size: int,
    requests: int,
    seed: int,
) -> None:
    random.seed(seed)

    contiguous = ContiguousAllocator(total_tokens)
    paged = PagedAllocator(total_tokens, block_size)

    active_requests: list[str] = []

    contiguous_failures = 0
    paged_failures = 0

    print("\n" + "=" * 72)
    print("RANDOMIZED SIMULATION")
    print("=" * 72)

    for step in range(1, requests + 1):
        should_free = active_requests and random.random() < 0.35

        if should_free:
            request_id = random.choice(active_requests)

            contiguous.free(request_id)
            paged.free(request_id)
            active_requests.remove(request_id)

            print(f"Step {step:>3}: freed {request_id}")
            continue

        request_id = f"req-{step}"
        token_count = random.randint(8, 64)

        contiguous_result = contiguous.allocate(
            request_id,
            token_count,
        )

        paged_result = paged.allocate(
            request_id,
            token_count,
        )

        if not contiguous_result.success:
            contiguous_failures += 1

        if not paged_result.success:
            paged_failures += 1

        if contiguous_result.success and paged_result.success:
            active_requests.append(request_id)
        else:
            if contiguous_result.success:
                contiguous.free(request_id)

            if paged_result.success:
                paged.free(request_id)

        print(
            f"Step {step:>3}: "
            f"{request_id:<8} tokens={token_count:<3} | "
            f"contiguous={'ok' if contiguous_result.success else 'fail':<4} | "
            f"paged={'ok' if paged_result.success else 'fail':<4}"
        )

    print("\n" + "=" * 72)
    print("SIMULATION SUMMARY")
    print("=" * 72)

    print(f"Total operations              : {requests}")
    print(f"Active requests               : {len(active_requests)}")
    print(f"Contiguous allocation failures: {contiguous_failures}")
    print(f"Paged allocation failures     : {paged_failures}")

    contiguous.print_state()
    paged.print_state()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare contiguous and paged KV-cache allocation."
        )
    )

    parser.add_argument(
        "--total-tokens",
        type=int,
        default=128,
        help="Total simulated KV-cache token capacity.",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Paged allocator block size in tokens.",
    )

    parser.add_argument(
        "--random-simulation",
        action="store_true",
        help="Run a randomized allocation/free simulation.",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=30,
        help="Number of randomized operations.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.total_tokens <= 0:
        raise ValueError("--total-tokens must be positive.")

    if args.block_size <= 0:
        raise ValueError("--block-size must be positive.")

    if args.total_tokens % args.block_size != 0:
        raise ValueError(
            "--total-tokens must be divisible by --block-size."
        )

    run_deterministic_demo(
        total_tokens=args.total_tokens,
        block_size=args.block_size,
    )

    if args.random_simulation:
        run_random_simulation(
            total_tokens=args.total_tokens,
            block_size=args.block_size,
            requests=args.requests,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
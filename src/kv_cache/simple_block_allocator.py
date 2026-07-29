import argparse
import math
from dataclasses import dataclass, field


@dataclass
class RequestAllocation:
    """Tracks cache blocks assigned to one request."""

    request_id: str
    token_count: int
    block_ids: list[int] = field(default_factory=list)


class KVBlockAllocator:
    """
    Simple fixed-size KV-cache block allocator.

    Each block stores KV data for a fixed number of tokens.
    """

    def __init__(
        self,
        total_blocks: int,
        block_size_tokens: int,
    ) -> None:
        if total_blocks <= 0:
            raise ValueError("total_blocks must be greater than zero.")

        if block_size_tokens <= 0:
            raise ValueError(
                "block_size_tokens must be greater than zero."
            )

        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens

        self.free_blocks: list[int] = list(range(total_blocks))
        self.allocations: dict[str, RequestAllocation] = {}

    def blocks_required(self, token_count: int) -> int:
        """Return the number of blocks required for a token count."""
        if token_count < 0:
            raise ValueError("token_count cannot be negative.")

        if token_count == 0:
            return 0

        return math.ceil(token_count / self.block_size_tokens)

    def allocate(
        self,
        request_id: str,
        token_count: int,
    ) -> RequestAllocation:
        """Allocate enough blocks for a new request."""
        if request_id in self.allocations:
            raise ValueError(
                f"Request '{request_id}' already has an allocation."
            )

        required_blocks = self.blocks_required(token_count)

        if required_blocks > len(self.free_blocks):
            raise MemoryError(
                f"Not enough free KV blocks for request '{request_id}'. "
                f"Required: {required_blocks}, "
                f"available: {len(self.free_blocks)}."
            )

        assigned_blocks = [
            self.free_blocks.pop(0)
            for _ in range(required_blocks)
        ]

        allocation = RequestAllocation(
            request_id=request_id,
            token_count=token_count,
            block_ids=assigned_blocks,
        )

        self.allocations[request_id] = allocation
        return allocation

    def append_tokens(
        self,
        request_id: str,
        additional_tokens: int,
    ) -> RequestAllocation:
        """
        Extend an existing request and allocate more blocks if needed.
        """
        if additional_tokens < 0:
            raise ValueError("additional_tokens cannot be negative.")

        if request_id not in self.allocations:
            raise KeyError(
                f"Request '{request_id}' does not exist."
            )

        allocation = self.allocations[request_id]

        old_token_count = allocation.token_count
        new_token_count = old_token_count + additional_tokens

        old_block_count = self.blocks_required(old_token_count)
        new_block_count = self.blocks_required(new_token_count)
        extra_blocks_required = new_block_count - old_block_count

        if extra_blocks_required > len(self.free_blocks):
            raise MemoryError(
                f"Cannot extend request '{request_id}'. "
                f"Additional blocks required: {extra_blocks_required}, "
                f"available: {len(self.free_blocks)}."
            )

        for _ in range(extra_blocks_required):
            allocation.block_ids.append(
                self.free_blocks.pop(0)
            )

        allocation.token_count = new_token_count
        return allocation

    def free(self, request_id: str) -> None:
        """Release all blocks assigned to a request."""
        if request_id not in self.allocations:
            raise KeyError(
                f"Request '{request_id}' does not exist."
            )

        allocation = self.allocations.pop(request_id)

        self.free_blocks.extend(allocation.block_ids)
        self.free_blocks.sort()

    def used_block_count(self) -> int:
        """Return the number of allocated blocks."""
        return self.total_blocks - len(self.free_blocks)

    def used_token_capacity(self) -> int:
        """Return token capacity represented by allocated blocks."""
        return self.used_block_count() * self.block_size_tokens

    def actual_cached_tokens(self) -> int:
        """Return the real number of tokens stored by all requests."""
        return sum(
            allocation.token_count
            for allocation in self.allocations.values()
        )

    def internal_fragmentation_tokens(self) -> int:
        """
        Return unused token slots inside allocated blocks.

        Example:
        block size = 16
        request tokens = 18
        allocated capacity = 32
        internal fragmentation = 14 token slots
        """
        return (
            self.used_token_capacity()
            - self.actual_cached_tokens()
        )

    def utilization_percent(self) -> float:
        """Return the percentage of total blocks currently allocated."""
        if self.total_blocks == 0:
            return 0.0

        return (
            self.used_block_count()
            / self.total_blocks
            * 100
        )

    def allocated_capacity_utilization_percent(self) -> float:
        """
        Return how efficiently allocated block capacity is being used.
        """
        capacity = self.used_token_capacity()

        if capacity == 0:
            return 0.0

        return (
            self.actual_cached_tokens()
            / capacity
            * 100
        )

    def print_state(self) -> None:
        """Print current allocator state."""
        print("\n" + "=" * 72)
        print("KV BLOCK ALLOCATOR STATE")
        print("=" * 72)

        print(f"Total blocks              : {self.total_blocks}")
        print(f"Free blocks               : {len(self.free_blocks)}")
        print(f"Used blocks               : {self.used_block_count()}")
        print(
            f"Block size                : "
            f"{self.block_size_tokens} tokens"
        )
        print(
            f"Total token capacity      : "
            f"{self.total_blocks * self.block_size_tokens}"
        )
        print(
            f"Allocated token capacity  : "
            f"{self.used_token_capacity()}"
        )
        print(
            f"Actual cached tokens      : "
            f"{self.actual_cached_tokens()}"
        )
        print(
            f"Internal fragmentation    : "
            f"{self.internal_fragmentation_tokens()} token slots"
        )
        print(
            f"Block utilization         : "
            f"{self.utilization_percent():.2f}%"
        )
        print(
            f"Allocated capacity usage  : "
            f"{self.allocated_capacity_utilization_percent():.2f}%"
        )

        print("\nActive requests:")

        if not self.allocations:
            print("  None")
        else:
            for request_id, allocation in self.allocations.items():
                allocated_capacity = (
                    len(allocation.block_ids)
                    * self.block_size_tokens
                )

                unused_slots = (
                    allocated_capacity
                    - allocation.token_count
                )

                print(
                    f"  {request_id}: "
                    f"tokens={allocation.token_count}, "
                    f"blocks={allocation.block_ids}, "
                    f"capacity={allocated_capacity}, "
                    f"unused={unused_slots}"
                )

        print(f"\nFree block IDs: {self.free_blocks}")


def run_demo(
    total_blocks: int,
    block_size_tokens: int,
) -> None:
    """Run a small allocator demonstration."""
    allocator = KVBlockAllocator(
        total_blocks=total_blocks,
        block_size_tokens=block_size_tokens,
    )

    print("=" * 72)
    print("SIMPLE PAGED KV-CACHE ALLOCATOR")
    print("=" * 72)

    allocator.print_state()

    print("\nAllocating request-A with 18 tokens...")
    allocation_a = allocator.allocate(
        request_id="request-A",
        token_count=18,
    )
    print(f"Assigned blocks: {allocation_a.block_ids}")
    allocator.print_state()

    print("\nAllocating request-B with 30 tokens...")
    allocation_b = allocator.allocate(
        request_id="request-B",
        token_count=30,
    )
    print(f"Assigned blocks: {allocation_b.block_ids}")
    allocator.print_state()

    print("\nAppending 20 tokens to request-A...")
    updated_a = allocator.append_tokens(
        request_id="request-A",
        additional_tokens=20,
    )
    print(f"Updated blocks: {updated_a.block_ids}")
    allocator.print_state()

    print("\nFreeing request-B...")
    allocator.free("request-B")
    allocator.print_state()

    print("\nAllocating request-C with 45 tokens...")
    allocation_c = allocator.allocate(
        request_id="request-C",
        token_count=45,
    )
    print(f"Assigned blocks: {allocation_c.block_ids}")
    allocator.print_state()

    print("\nTrying an allocation larger than available memory...")

    try:
        allocator.allocate(
            request_id="request-too-large",
            token_count=100_000,
        )
    except MemoryError as error:
        print(f"Expected allocation failure: {error}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate fixed-size block allocation for an "
            "educational KV cache."
        )
    )

    parser.add_argument(
        "--total-blocks",
        type=int,
        default=12,
        help="Total number of KV-cache blocks.",
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Number of tokens stored in each block.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    run_demo(
        total_blocks=args.total_blocks,
        block_size_tokens=args.block_size,
    )


if __name__ == "__main__":
    main()
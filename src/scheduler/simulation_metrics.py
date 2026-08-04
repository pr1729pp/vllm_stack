import math
import statistics
from dataclasses import dataclass

from request import Request
from simulation import SimulationResult


@dataclass
class InferenceStatistics:
    """Summary statistics for an inference simulation."""

    completed_requests: int
    total_steps: int
    total_generated_tokens: int

    average_waiting_time: float
    maximum_waiting_time: int

    average_completion_time: float
    maximum_completion_time: int

    p50_completion_time: float
    p95_completion_time: float
    p99_completion_time: float

    request_throughput: float
    token_throughput: float

    average_batch_size: float
    batch_slot_utilization_percent: float

    average_queue_length: float
    maximum_queue_length: int


def percentile(
    values: list[int | float],
    percentile_value: float,
) -> float:
    """
    Calculate a percentile using linear interpolation.

    percentile_value must be between 0 and 100.
    """
    if not values:
        return 0.0

    if not 0 <= percentile_value <= 100:
        raise ValueError(
            "percentile_value must be between 0 and 100."
        )

    sorted_values = sorted(float(value) for value in values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (
        percentile_value / 100
        * (len(sorted_values) - 1)
    )

    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    interpolation_weight = position - lower_index

    return (
        lower_value
        + (upper_value - lower_value)
        * interpolation_weight
    )


def calculate_statistics(
    result: SimulationResult,
) -> InferenceStatistics:
    """Calculate inference-system metrics from simulation output."""
    completed_requests = result.completed_requests
    step_records = result.step_records

    waiting_times = [
        request.waiting_time
        for request in completed_requests
    ]

    completion_times = [
        request.completion_time
        for request in completed_requests
    ]

    queue_lengths = [
        record.waiting_count
        for record in step_records
    ]

    completed_count = len(completed_requests)

    request_throughput = (
        completed_count / result.total_steps
        if result.total_steps > 0
        else 0.0
    )

    token_throughput = (
        result.total_generated_tokens / result.total_steps
        if result.total_steps > 0
        else 0.0
    )

    return InferenceStatistics(
        completed_requests=completed_count,
        total_steps=result.total_steps,
        total_generated_tokens=result.total_generated_tokens,

        average_waiting_time=(
            statistics.mean(waiting_times)
            if waiting_times
            else 0.0
        ),
        maximum_waiting_time=(
            max(waiting_times)
            if waiting_times
            else 0
        ),

        average_completion_time=(
            statistics.mean(completion_times)
            if completion_times
            else 0.0
        ),
        maximum_completion_time=(
            max(completion_times)
            if completion_times
            else 0
        ),

        p50_completion_time=percentile(
            completion_times,
            50,
        ),
        p95_completion_time=percentile(
            completion_times,
            95,
        ),
        p99_completion_time=percentile(
            completion_times,
            99,
        ),

        request_throughput=request_throughput,
        token_throughput=token_throughput,

        average_batch_size=result.average_batch_size,
        batch_slot_utilization_percent=(
            result.batch_slot_utilization_percent
        ),

        average_queue_length=(
            statistics.mean(queue_lengths)
            if queue_lengths
            else 0.0
        ),
        maximum_queue_length=(
            max(queue_lengths)
            if queue_lengths
            else 0
        ),
    )


def print_statistics(
    metrics: InferenceStatistics,
) -> None:
    """Print inference metrics in a readable format."""
    print("\n" + "=" * 76)
    print("INFERENCE PERFORMANCE STATISTICS")
    print("=" * 76)

    print("\nWork completed")
    print(
        f"Completed requests        : "
        f"{metrics.completed_requests}"
    )
    print(
        f"Simulation steps          : "
        f"{metrics.total_steps}"
    )
    print(
        f"Generated output tokens   : "
        f"{metrics.total_generated_tokens}"
    )

    print("\nLatency")
    print(
        f"Average waiting time      : "
        f"{metrics.average_waiting_time:.2f} steps"
    )
    print(
        f"Maximum waiting time      : "
        f"{metrics.maximum_waiting_time} steps"
    )
    print(
        f"Average completion time   : "
        f"{metrics.average_completion_time:.2f} steps"
    )
    print(
        f"Maximum completion time   : "
        f"{metrics.maximum_completion_time} steps"
    )
    print(
        f"P50 completion time       : "
        f"{metrics.p50_completion_time:.2f} steps"
    )
    print(
        f"P95 completion time       : "
        f"{metrics.p95_completion_time:.2f} steps"
    )
    print(
        f"P99 completion time       : "
        f"{metrics.p99_completion_time:.2f} steps"
    )

    print("\nThroughput")
    print(
        f"Request throughput        : "
        f"{metrics.request_throughput:.4f} requests/step"
    )
    print(
        f"Token throughput          : "
        f"{metrics.token_throughput:.2f} tokens/step"
    )

    print("\nScheduler utilization")
    print(
        f"Average batch size        : "
        f"{metrics.average_batch_size:.2f}"
    )
    print(
        f"Batch-slot utilization    : "
        f"{metrics.batch_slot_utilization_percent:.2f}%"
    )
    print(
        f"Average waiting queue     : "
        f"{metrics.average_queue_length:.2f}"
    )
    print(
        f"Maximum waiting queue     : "
        f"{metrics.maximum_queue_length}"
    )


def print_request_statistics(
    requests: list[Request],
) -> None:
    """Print lifecycle metrics for every completed request."""
    print("\n" + "=" * 88)
    print("PER-REQUEST STATISTICS")
    print("=" * 88)

    print(
        f"{'Request':<14}"
        f"{'Arrival':>9}"
        f"{'Start':>9}"
        f"{'Finish':>9}"
        f"{'Wait':>9}"
        f"{'Output':>10}"
        f"{'Completion':>14}"
    )

    print("-" * 88)

    ordered_requests = sorted(
        requests,
        key=lambda request: (
            request.arrival_step,
            request.request_id,
        ),
    )

    for request in ordered_requests:
        print(
            f"{request.request_id:<14}"
            f"{request.arrival_step:>9}"
            f"{request.start_step:>9}"
            f"{request.finish_step:>9}"
            f"{request.waiting_time:>9}"
            f"{request.output_tokens:>10}"
            f"{request.completion_time:>14}"
        )
import argparse
from dataclasses import dataclass, field

from request import Request
from request_generator import RequestGenerator, WorkloadConfig
from scheduler import (
    ContinuousBatchScheduler,
    SchedulerStepResult,
)


@dataclass
class SimulationStepRecord:
    """Metrics collected during one simulation step."""

    step: int
    arrived_request_ids: list[str] = field(default_factory=list)
    admitted_request_ids: list[str] = field(default_factory=list)
    running_request_ids: list[str] = field(default_factory=list)
    completed_request_ids: list[str] = field(default_factory=list)

    waiting_count: int = 0
    running_count: int = 0
    completed_count: int = 0
    generated_tokens: int = 0


@dataclass
class SimulationResult:
    """Complete output from one simulation run."""

    completed_requests: list[Request]
    step_records: list[SimulationStepRecord]
    total_steps: int
    total_generated_tokens: int
    average_batch_size: float
    batch_slot_utilization_percent: float


class ContinuousBatchingSimulation:
    """
    Event-loop simulation for continuous batching.

    During each step:

    1. Requests whose arrival time has been reached enter the server.
    2. The scheduler moves waiting requests into free batch slots.
    3. Every running request generates one token.
    4. Finished requests leave the active batch.
    5. Step-level metrics are recorded.
    """

    def __init__(
        self,
        requests: list[Request],
        maximum_batch_size: int,
        verbose: bool = True,
    ) -> None:
        if maximum_batch_size <= 0:
            raise ValueError(
                "maximum_batch_size must be greater than zero."
            )

        self.pending_arrivals = sorted(
            requests,
            key=lambda request: (
                request.arrival_step,
                request.request_id,
            ),
        )

        self.scheduler = ContinuousBatchScheduler(
            maximum_batch_size=maximum_batch_size,
        )

        self.verbose = verbose
        self.current_step = 0
        self.step_records: list[SimulationStepRecord] = []

    def _collect_arrivals(self) -> list[Request]:
        """
        Remove and return requests that have arrived by this step.
        """
        arrived_requests: list[Request] = []

        while (
            self.pending_arrivals
            and self.pending_arrivals[0].arrival_step
            <= self.current_step
        ):
            arrived_requests.append(
                self.pending_arrivals.pop(0)
            )

        return arrived_requests

    def _has_remaining_work(self) -> bool:
        """Return True while any request is pending, waiting, or running."""
        return bool(
            self.pending_arrivals
            or not self.scheduler.is_idle()
        )

    def _skip_to_next_arrival_if_idle(self) -> None:
        """
        Move directly to the next arrival when the server is completely idle.

        This avoids producing many empty simulation steps when there is a large
        gap between request arrivals.
        """
        if not self.pending_arrivals:
            return

        if not self.scheduler.is_idle():
            return

        next_arrival_step = self.pending_arrivals[0].arrival_step

        if next_arrival_step > self.current_step:
            if self.verbose:
                print(
                    f"\nServer idle at step {self.current_step}. "
                    f"Advancing to step {next_arrival_step}."
                )

            self.current_step = next_arrival_step

    def _create_step_record(
        self,
        arrived_requests: list[Request],
        scheduler_result: SchedulerStepResult,
    ) -> SimulationStepRecord:
        """Create a record for the completed simulation step."""
        return SimulationStepRecord(
            step=self.current_step,
            arrived_request_ids=[
                request.request_id
                for request in arrived_requests
            ],
            admitted_request_ids=(
                scheduler_result.admitted_request_ids
            ),
            running_request_ids=(
                scheduler_result.running_request_ids
            ),
            completed_request_ids=(
                scheduler_result.completed_request_ids
            ),
            waiting_count=self.scheduler.waiting_count(),
            running_count=self.scheduler.running_count(),
            completed_count=self.scheduler.completed_count(),
            generated_tokens=scheduler_result.generated_tokens,
        )

    def _print_step(
        self,
        record: SimulationStepRecord,
    ) -> None:
        """Print a readable summary of one simulation step."""
        print("\n" + "=" * 76)
        print(f"SIMULATION STEP {record.step}")
        print("=" * 76)

        print(
            "Arrived     : "
            + (
                ", ".join(record.arrived_request_ids)
                if record.arrived_request_ids
                else "None"
            )
        )

        print(
            "Admitted    : "
            + (
                ", ".join(record.admitted_request_ids)
                if record.admitted_request_ids
                else "None"
            )
        )

        print(
            "Ran this step: "
            + (
                ", ".join(record.running_request_ids)
                if record.running_request_ids
                else "None"
            )
        )

        print(
            "Completed   : "
            + (
                ", ".join(record.completed_request_ids)
                if record.completed_request_ids
                else "None"
            )
        )

        print(
            f"Generated tokens this step: "
            f"{record.generated_tokens}"
        )

        print(
            f"Queue state: "
            f"waiting={record.waiting_count}, "
            f"running={record.running_count}, "
            f"completed={record.completed_count}"
        )

    def run(self) -> SimulationResult:
        """Run until all generated requests have completed."""
        print("=" * 76)
        print("CONTINUOUS BATCHING SIMULATION")
        print("=" * 76)

        print(
            f"Requests           : {len(self.pending_arrivals)}"
        )
        print(
            f"Maximum batch size : "
            f"{self.scheduler.maximum_batch_size}"
        )

        while self._has_remaining_work():
            self._skip_to_next_arrival_if_idle()

            arrived_requests = self._collect_arrivals()

            if arrived_requests:
                self.scheduler.add_requests(arrived_requests)

            scheduler_result = self.scheduler.decode_one_step(
                step=self.current_step,
            )

            record = self._create_step_record(
                arrived_requests=arrived_requests,
                scheduler_result=scheduler_result,
            )

            self.step_records.append(record)

            if self.verbose:
                self._print_step(record)

            self.current_step += 1

        result = SimulationResult(
            completed_requests=(
                self.scheduler.completed_requests.copy()
            ),
            step_records=self.step_records.copy(),
            total_steps=self.scheduler.scheduler_steps,
            total_generated_tokens=(
                self.scheduler.total_generated_tokens
            ),
            average_batch_size=(
                self.scheduler.average_batch_size()
            ),
            batch_slot_utilization_percent=(
                self.scheduler.batch_slot_utilization_percent()
            ),
        )

        self._print_final_summary(result)

        return result

    @staticmethod
    def _print_final_summary(
        result: SimulationResult,
    ) -> None:
        """Print basic metrics after all requests complete."""
        print("\n" + "=" * 76)
        print("SIMULATION COMPLETED")
        print("=" * 76)

        print(
            f"Completed requests      : "
            f"{len(result.completed_requests)}"
        )
        print(
            f"Decode iterations       : "
            f"{result.total_steps}"
        )
        print(
            f"Generated output tokens : "
            f"{result.total_generated_tokens}"
        )
        print(
            f"Average batch size      : "
            f"{result.average_batch_size:.2f}"
        )
        print(
            f"Batch-slot utilization  : "
            f"{result.batch_slot_utilization_percent:.2f}%"
        )


def print_workload(requests: list[Request]) -> None:
    """Print the workload before beginning the simulation."""
    print("\n" + "=" * 76)
    print("WORKLOAD")
    print("=" * 76)

    print(
        f"{'Request':<14}"
        f"{'Arrival':>10}"
        f"{'Prompt':>12}"
        f"{'Output':>12}"
    )

    print("-" * 76)

    for request in requests:
        print(
            f"{request.request_id:<14}"
            f"{request.arrival_step:>10}"
            f"{request.prompt_tokens:>12}"
            f"{request.output_tokens:>12}"
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line simulation settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a continuously batched LLM inference server."
        )
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=12,
        help="Number of inference requests.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum active decode batch size.",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="bursty",
        choices=sorted(
            RequestGenerator.SUPPORTED_PATTERNS
        ),
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
        default=24,
        help="Maximum generated output length.",
    )

    parser.add_argument(
        "--max-arrival-step",
        type=int,
        default=15,
        help="Maximum arrival step for random traffic.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide individual simulation steps.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    workload_config = WorkloadConfig(
        number_of_requests=args.requests,
        arrival_pattern=args.pattern,
        minimum_prompt_tokens=args.min_prompt,
        maximum_prompt_tokens=args.max_prompt,
        minimum_output_tokens=args.min_output,
        maximum_output_tokens=args.max_output,
        maximum_arrival_step=args.max_arrival_step,
        seed=args.seed,
    )

    request_generator = RequestGenerator(
        config=workload_config,
    )

    requests = request_generator.generate()

    print_workload(requests)

    simulation = ContinuousBatchingSimulation(
        requests=requests,
        maximum_batch_size=args.batch_size,
        verbose=not args.quiet,
    )

    #simulation.run()
    result = simulation.run()

    from simulation_metrics import (
        calculate_statistics,
        print_request_statistics,
        print_statistics,
    )

    metrics = calculate_statistics(result)

    print_statistics(metrics)
    print_request_statistics(result.completed_requests)


if __name__ == "__main__":
    main()

from collections import deque
from dataclasses import dataclass, field

from request import Request


@dataclass
class SchedulerStepResult:
    """Summary of one scheduler iteration."""

    step: int
    admitted_request_ids: list[str] = field(default_factory=list)
    running_request_ids: list[str] = field(default_factory=list)
    completed_request_ids: list[str] = field(default_factory=list)
    generated_tokens: int = 0


class ContinuousBatchScheduler:
    """
    Educational continuous-batching scheduler.

    At every simulation step, the scheduler:

    1. Accepts newly arrived requests.
    2. Fills available batch slots.
    3. Generates one token for each running request.
    4. Removes completed requests.
    5. Makes the freed slots available for the next step.
    """

    def __init__(self, maximum_batch_size: int) -> None:
        if maximum_batch_size <= 0:
            raise ValueError(
                "maximum_batch_size must be greater than zero."
            )

        self.maximum_batch_size = maximum_batch_size

        self.waiting_queue: deque[Request] = deque()
        self.running_requests: list[Request] = []
        self.completed_requests: list[Request] = []

        self.total_generated_tokens = 0
        self.total_batch_slots_used = 0
        self.scheduler_steps = 0

    def add_request(self, request: Request) -> None:
        """Add one request to the waiting queue."""
        if request.finished:
            raise ValueError(
                f"Request '{request.request_id}' is already finished."
            )

        if self._contains_request_id(request.request_id):
            raise ValueError(
                f"Duplicate request ID: '{request.request_id}'."
            )

        self.waiting_queue.append(request)

    def add_requests(self, requests: list[Request]) -> None:
        """Add several requests to the waiting queue."""
        for request in requests:
            self.add_request(request)

    def _contains_request_id(self, request_id: str) -> bool:
        """Check every scheduler queue for an existing request ID."""
        all_requests = (
            list(self.waiting_queue)
            + self.running_requests
            + self.completed_requests
        )

        return any(
            request.request_id == request_id
            for request in all_requests
        )

    def available_batch_slots(self) -> int:
        """Return the number of unused running-batch positions."""
        return (
            self.maximum_batch_size
            - len(self.running_requests)
        )

    def admit_waiting_requests(self, step: int) -> list[str]:
        """
        Move waiting requests into currently available batch slots.

        Requests are scheduled in first-in, first-out order.
        """
        admitted_request_ids: list[str] = []

        while (
            self.waiting_queue
            and self.available_batch_slots() > 0
        ):
            request = self.waiting_queue.popleft()

            if request.start_step is None:
                request.start_step = step

            self.running_requests.append(request)
            admitted_request_ids.append(request.request_id)

        return admitted_request_ids

    def decode_one_step(
        self,
        step: int,
    ) -> SchedulerStepResult:
        """
        Generate one token for every currently running request.

        A simulation step represents one decode iteration, not one real
        second. Every active request advances by exactly one output token.
        """
        admitted_request_ids = self.admit_waiting_requests(step)

        running_at_start = [
            request.request_id
            for request in self.running_requests
        ]

        completed_request_ids: list[str] = []
        generated_tokens = 0

        for request in self.running_requests:
            request.generated_tokens += 1
            generated_tokens += 1

            if request.finished:
                # The request generated its final token during this step.
                request.finish_step = step + 1
                completed_request_ids.append(request.request_id)

        if completed_request_ids:
            completed_id_set = set(completed_request_ids)

            still_running: list[Request] = []

            for request in self.running_requests:
                if request.request_id in completed_id_set:
                    self.completed_requests.append(request)
                else:
                    still_running.append(request)

            self.running_requests = still_running

        self.total_generated_tokens += generated_tokens
        self.total_batch_slots_used += len(running_at_start)
        self.scheduler_steps += 1

        return SchedulerStepResult(
            step=step,
            admitted_request_ids=admitted_request_ids,
            running_request_ids=running_at_start,
            completed_request_ids=completed_request_ids,
            generated_tokens=generated_tokens,
        )

    def is_idle(self) -> bool:
        """Return True when no request is waiting or running."""
        return (
            not self.waiting_queue
            and not self.running_requests
        )

    def waiting_count(self) -> int:
        """Return the number of queued requests."""
        return len(self.waiting_queue)

    def running_count(self) -> int:
        """Return the number of currently active requests."""
        return len(self.running_requests)

    def completed_count(self) -> int:
        """Return the number of finished requests."""
        return len(self.completed_requests)

    def average_batch_size(self) -> float:
        """Return the average number of active requests per decode step."""
        if self.scheduler_steps == 0:
            return 0.0

        return (
            self.total_batch_slots_used
            / self.scheduler_steps
        )

    def batch_slot_utilization_percent(self) -> float:
        """
        Return average utilization of the configured batch capacity.
        """
        total_possible_slots = (
            self.scheduler_steps
            * self.maximum_batch_size
        )

        if total_possible_slots == 0:
            return 0.0

        return (
            self.total_batch_slots_used
            / total_possible_slots
            * 100
        )

    def print_state(self, step: int) -> None:
        """Print the current state of every scheduler queue."""
        print("\n" + "=" * 72)
        print(f"SCHEDULER STATE — STEP {step}")
        print("=" * 72)

        print(
            f"Maximum batch size : {self.maximum_batch_size}"
        )
        print(
            f"Waiting requests   : {self.waiting_count()}"
        )
        print(
            f"Running requests   : {self.running_count()}"
        )
        print(
            f"Completed requests : {self.completed_count()}"
        )

        print("\nWaiting queue:")

        if not self.waiting_queue:
            print("  None")
        else:
            for request in self.waiting_queue:
                print(
                    f"  {request.request_id}: "
                    f"arrival={request.arrival_step}, "
                    f"output={request.output_tokens}"
                )

        print("\nRunning batch:")

        if not self.running_requests:
            print("  None")
        else:
            for request in self.running_requests:
                print(
                    f"  {request.request_id}: "
                    f"generated={request.generated_tokens}/"
                    f"{request.output_tokens}"
                )

        print("\nCompleted requests:")

        if not self.completed_requests:
            print("  None")
        else:
            for request in self.completed_requests:
                print(
                    f"  {request.request_id}: "
                    f"start={request.start_step}, "
                    f"finish={request.finish_step}"
                )


def print_step_result(result: SchedulerStepResult) -> None:
    """Print a concise summary of one scheduler step."""
    print(f"\nStep {result.step}")

    print(
        "  Admitted  : "
        + (
            ", ".join(result.admitted_request_ids)
            if result.admitted_request_ids
            else "None"
        )
    )

    print(
        "  Running   : "
        + (
            ", ".join(result.running_request_ids)
            if result.running_request_ids
            else "None"
        )
    )

    print(
        "  Completed : "
        + (
            ", ".join(result.completed_request_ids)
            if result.completed_request_ids
            else "None"
        )
    )

    print(
        f"  Tokens generated this step: "
        f"{result.generated_tokens}"
    )


def run_demo() -> None:
    """Run a deterministic continuous-batching example."""
    requests = [
        Request(
            request_id="A",
            arrival_step=0,
            prompt_tokens=100,
            output_tokens=6,
        ),
        Request(
            request_id="B",
            arrival_step=0,
            prompt_tokens=40,
            output_tokens=2,
        ),
        Request(
            request_id="C",
            arrival_step=0,
            prompt_tokens=80,
            output_tokens=4,
        ),
        Request(
            request_id="D",
            arrival_step=0,
            prompt_tokens=30,
            output_tokens=3,
        ),
        Request(
            request_id="E",
            arrival_step=0,
            prompt_tokens=60,
            output_tokens=2,
        ),
    ]

    scheduler = ContinuousBatchScheduler(
        maximum_batch_size=3
    )

    scheduler.add_requests(requests)

    print("=" * 72)
    print("CONTINUOUS BATCHING SCHEDULER DEMO")
    print("=" * 72)

    step = 0

    while not scheduler.is_idle():
        result = scheduler.decode_one_step(step)
        print_step_result(result)
        step += 1

    print("\n" + "=" * 72)
    print("FINAL METRICS")
    print("=" * 72)

    print(
        f"Completed requests      : "
        f"{scheduler.completed_count()}"
    )
    print(
        f"Generated tokens        : "
        f"{scheduler.total_generated_tokens}"
    )
    print(
        f"Scheduler steps         : "
        f"{scheduler.scheduler_steps}"
    )
    print(
        f"Average batch size      : "
        f"{scheduler.average_batch_size():.2f}"
    )
    print(
        f"Batch-slot utilization  : "
        f"{scheduler.batch_slot_utilization_percent():.2f}%"
    )


if __name__ == "__main__":
    run_demo()
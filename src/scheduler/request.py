from dataclasses import dataclass


@dataclass
class Request:
    request_id: str
    arrival_step: int

    prompt_tokens: int
    output_tokens: int

    generated_tokens: int = 0

    start_step: int | None = None
    finish_step: int | None = None

    @property
    def finished(self) -> bool:
        return self.generated_tokens >= self.output_tokens

    @property
    def waiting_time(self) -> int:
        if self.start_step is None:
            return 0

        return self.start_step - self.arrival_step

    @property
    def completion_time(self) -> int:
        if self.finish_step is None:
            return 0

        return self.finish_step - self.arrival_step
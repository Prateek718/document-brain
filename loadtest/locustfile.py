"""Load test for the Document Brain query endpoint.

Run against a container whose ANTHROPIC_BASE_URL points at the fake
Anthropic server, so we measure our own service under concurrency
without real token spend.
"""

import random
from typing import ClassVar

from locust import HttpUser, LoadTestShape, between, task

# Questions should match the ingested document so retrieval returns real
# chunks and we exercise the full prompt-building path, not the empty-context
# shortcut. Edit these to fit whatever PDF is in your collection.
_QUESTIONS = [
    "What is the difference between accuracy and precision in measurement?",
    "Explain significant figures and the rules for counting them.",
    "What are systematic and random errors?",
    "How is dimensional analysis used to check the correctness of an equation?",
    "What are the SI base units and their definitions?",
    "Explain the difference between distance and displacement.",
    "What is the difference between average velocity and instantaneous velocity?",
    "Derive the equations of motion for uniformly accelerated motion.",
    "How do you interpret a position-time graph for an object at rest versus in motion?",
    "What is relative velocity and how is it calculated for objects moving along a straight line?",
]


class DocumentBrainUser(HttpUser):
    # Wait 1-3s between requests to imitate a human reading the answer
    # before asking again, rather than hammering with zero think-time.
    wait_time = between(1, 3)

    @task
    def ask_question(self) -> None:
        question = random.choice(_QUESTIONS)
        # name= groups all questions under one entry in Locust's stats
        # instead of splitting metrics per unique question string.
        self.client.post(
            "/query",
            json={"question": question},
            name="/query",
        )


class StagedRampShape(LoadTestShape):
    """Steps concurrency up in holds so percentiles stabilize at each level,
    making the saturation knee visible. Each stage: cumulative seconds, users, spawn rate."""

    stages: ClassVar[list[dict[str, int]]] = [
        {"duration": 60, "users": 10, "spawn_rate": 2},
        {"duration": 120, "users": 20, "spawn_rate": 2},
        {"duration": 180, "users": 30, "spawn_rate": 2},
        {"duration": 240, "users": 40, "spawn_rate": 2},
        {"duration": 300, "users": 50, "spawn_rate": 2},
        {"duration": 360, "users": 60, "spawn_rate": 2},
        {"duration": 420, "users": 80, "spawn_rate": 4},
        {"duration": 480, "users": 100, "spawn_rate": 4},
    ]

    def tick(self) -> tuple[int, int] | None:
        run_time = self.get_run_time()  # type: ignore[no-untyped-call]  # Locust base method is untyped
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
        return None

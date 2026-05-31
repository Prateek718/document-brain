"""Load test for the Document Brain query endpoint.

Run against a container whose ANTHROPIC_BASE_URL points at the fake
Anthropic server, so we measure our own service under concurrency
without real token spend.
"""

import random

from locust import HttpUser, between, task

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

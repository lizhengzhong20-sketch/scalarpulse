"""Framework-independent ScalarPulse example."""

import math
import random
import time

import scalarpulse


run = scalarpulse.init(
    project="quickstart",
    name="tiny-classifier",
    config={"epochs": 3, "batch_size": 32},
    open_browser=True,
)

randomizer = random.Random(11)
try:
    for step in range(120):
        progress = step / 119
        run.log(
            {
                "train": {
                    "loss": 1.8 * math.exp(-4 * progress) + randomizer.uniform(-0.03, 0.03),
                    "accuracy": min(0.98, 0.35 + 0.65 * (1 - math.exp(-4 * progress))),
                },
                "learning_rate": 3e-3 * (1 - progress),
            },
            step=step,
        )
        time.sleep(0.08)
except KeyboardInterrupt:
    run.finish("stopped")
else:
    run.finish()

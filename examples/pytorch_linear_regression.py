"""A complete PyTorch example; PyTorch is intentionally not a ScalarPulse dependency."""

import torch

import scalarpulse
from scalarpulse.torch import TorchLogger


torch.manual_seed(7)
x = torch.linspace(-1, 1, 512).unsqueeze(1)
y = 2.5 * x - 0.4 + 0.12 * torch.randn_like(x)

model = torch.nn.Linear(1, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
criterion = torch.nn.MSELoss()

run = scalarpulse.init(
    project="regression",
    name="linear-adamw",
    framework="pytorch",
    config={"learning_rate": 0.03, "samples": len(x)},
    open_browser=True,
)
logger = TorchLogger(run, model=model, optimizer=optimizer, track_grad_norm=True)

try:
    for step in range(240):
        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        logger.log(loss=loss, step=step)
        optimizer.step()
except KeyboardInterrupt:
    run.finish("stopped")
else:
    run.finish()

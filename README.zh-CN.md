<div align="center">

# ScalarPulse

**轻量、零运行时依赖的模型训练指标实时看板。**

[![CI](https://github.com/lizhengzhong20-sketch/scalarpulse/actions/workflows/tests.yml/badge.svg)](https://github.com/lizhengzhong20-sketch/scalarpulse/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)

[English](README.md)

</div>

![代表 ScalarPulse 的脉冲与训练曲线](assets/scalarpulse-banner.png)

![ScalarPulse 实时展示 loss、准确率、学习率和吞吐量曲线](assets/dashboard.jpg)

ScalarPulse 是一个小型、可安装的 Python 包，用本地浏览器实时查看模型训练中的 loss、准确率、学习率、梯度范数以及任意标量指标。

- 核心运行时零第三方依赖；不会导入或绑定某个深度学习框架。
- PyTorch 优先：可直接记录 Tensor 标量、optimizer 学习率与模型梯度范数。
- 实时更新：训练写入 JSONL，浏览器通过 SSE 接收增量事件。
- 数据完全保存在本地，训练进程结束后仍能重新打开历史 run。
- 一个命令启动服务，也可以由 `scalarpulse.init()` 随训练自动启动。

> **预发布说明：** ScalarPulse 目前还没有发布到 PyPI。首个正式包发布前，请直接从这个 GitHub 仓库安装。

## 30 秒体验

安装当前 GitHub 版本：

```bash
python -m pip install "git+https://github.com/lizhengzhong20-sketch/scalarpulse.git"
scalarpulse demo --open
```

如果不希望自动打开浏览器，去掉 `--open`，再访问终端显示的地址（默认 `http://127.0.0.1:8765`）。

## 最小接入

```python
import scalarpulse

run = scalarpulse.init(
    project="mnist",
    name="cnn-baseline",
    config={"batch_size": 64, "learning_rate": 3e-4},
    open_browser=True,
)

for step in range(1_000):
    loss = train_one_step()
    accuracy = evaluate_if_needed()

    run.log(
        {
            "train": {"loss": loss, "accuracy": accuracy},
            "learning_rate": optimizer.param_groups[0]["lr"],
        },
        step=step,
    )

run.finish()
```

指标字典可以嵌套，ScalarPulse 会自动展开为 `train/loss`、`train/accuracy` 等曲线。Python 数值、NumPy 标量和 PyTorch 单元素 Tensor 都可以直接记录；非标量、`NaN` 和无穷值会被明确拒绝，避免写出浏览器无法解析的 JSON。

## PyTorch 辅助接口

```python
import scalarpulse
from scalarpulse.torch import TorchLogger

run = scalarpulse.init(project="cifar10", framework="pytorch")
logger = TorchLogger(
    run,
    model=model,
    optimizer=optimizer,
    track_grad_norm=True,
)

for step, (images, targets) in enumerate(loader):
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward()

    # 在 zero_grad() 之前调用，才能读取本 step 的梯度范数。
    logger.log(loss=loss, metrics={"accuracy": accuracy(logits, targets)}, step=step)
    optimizer.step()

run.finish()
```

`scalarpulse.torch` 采用 duck typing，本身不会 `import torch`，因此 ScalarPulse 核心可以在未安装 PyTorch 的环境中正常工作。

## 推荐的长时间训练方式

将仪表盘与训练进程分开运行，训练崩溃或重启时看板仍然保持：

```bash
scalarpulse serve --log-dir ./runs --open
```

```python
run = scalarpulse.init(log_dir="./runs", launch=False, project="experiment")
```

默认仅监听 `127.0.0.1`。如果改成 `0.0.0.0` 暴露到网络，请在受信任网络或反向代理的认证之后使用；当前 MVP 不包含账户认证。

## Python API

### `scalarpulse.init(...)` / `Tracker(...)`

主要参数：

- `project`、`name`：组织和标识 run。
- `config`、`tags`、`framework`：写入 run 元信息。
- `log_dir`：本地日志目录，默认 `.scalarpulse`。
- `launch`：是否随 Tracker 启动本地看板，默认开启。
- `host`、`port`、`open_browser`：本地服务设置。
- `quiet`：关闭启动提示。

### `run.log(metrics, step=None)`

追加一组标量。未传 `step` 时从 0 自动递增；显式 step 可以重复，方便在同一步分别记录训练与验证指标。

### `run.finish(status="completed")`

将 run 标记为 `completed`、`failed` 或 `stopped`，并保存最后一组指标摘要。重复调用是安全的，完成后继续 `log()` 会报错。

### `TorchLogger`

`TorchLogger.log()` 可以合并记录 loss、自定义指标、所有 optimizer 参数组的学习率，以及可选的全局梯度范数。

## 命令行

```text
scalarpulse serve [--log-dir DIR] [--host HOST] [--port PORT] [--open]
scalarpulse demo  [--steps N] [--interval SECONDS] [同样的服务参数]
```

## 日志格式

每个 run 都是一个独立目录：

```text
.scalarpulse/
└── 20260824-161500-a1b2c3/
    ├── meta.json
    └── metrics.jsonl
```

一行代表一次原子化的 `log()` 调用。这个格式便于追加、人工查看、版本迁移和后续导入 pandas。

## 开发与验证

```bash
git clone https://github.com/lizhengzhong20-sketch/scalarpulse.git
cd scalarpulse
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

当前版本定位为轻量 MVP：专注本地标量曲线，不包含云同步、分布式多进程合并、身份认证、图片/直方图或超大规模日志索引。这些可以在保持核心 API 稳定的前提下逐步加入。

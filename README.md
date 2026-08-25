<div align="center">

# ScalarPulse

**训练可以慢，指标不能失联。**

给它一个数，它就还你一条实时曲线。

[![CI](https://github.com/lizhengzhong20-sketch/scalarpulse/actions/workflows/tests.yml/badge.svg)](https://github.com/lizhengzhong20-sketch/scalarpulse/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/scalarpulse?color=3776AB&logo=pypi&logoColor=white)](https://pypi.org/project/scalarpulse/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](https://github.com/lizhengzhong20-sketch/scalarpulse/blob/main/LICENSE)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-6366f1.svg)](#项目边界)

[English](https://github.com/lizhengzhong20-sketch/scalarpulse/blob/main/README.en.md) · [30 秒体验](#30-秒体验) · [适用场景](#哪些模型和算法可以用) · [安全说明](https://github.com/lizhengzhong20-sketch/scalarpulse/blob/main/SECURITY.md)

</div>

![代表 ScalarPulse 的脉冲与训练曲线](https://raw.githubusercontent.com/lizhengzhong20-sketch/scalarpulse/main/assets/scalarpulse-banner.png)

ScalarPulse 是一个轻量、本地优先的 Python 训练指标看板。它把 loss、F1、学习率、梯度范数、延迟、Token 用量以及任何有限标量实时画进浏览器，并把每次实验保存成可读的 JSONL 文件。

它不负责训练模型，也不绑定某一种框架：**模型负责学习，ScalarPulse 负责把它的心情画出来。**

## 先看效果

![ScalarPulse 实时展示 loss、准确率、学习率和吞吐量曲线](https://raw.githubusercontent.com/lizhengzhong20-sketch/scalarpulse/main/assets/dashboard.jpg)

一个页面里可以查看运行状态、最新 step、指标数量和持续时间；切换历史 run、平滑曲线、暂停实时刷新、悬停读数、切换明暗主题，并导出当前 run 的 CSV。

## 为什么用它

- **接入很小**：在循环里调用一次 `run.log(...)`，不需要改造训练框架。
- **核心依赖为零**：运行时只使用 Python 标准库，适合教学、小实验和本地开发。
- **对 Python 生态友好**：普通 Python、NumPy、PyTorch、scikit-learn、XGBoost 都能记录。
- **数据留在本地**：无需账号、云服务或遥测；日志直接保存在你的目录中。
- **训练后仍能回看**：重新启动看板即可浏览已写入的历史 run。

## 30 秒体验

需要 Python 3.10 或更高版本：

```bash
python -m pip install scalarpulse
python -m scalarpulse demo --open
```

命令会生成一段模拟训练数据，并打开本地看板，默认地址为 `http://127.0.0.1:8765`。演示完成后看板会继续运行，按 `Ctrl+C` 退出。

如果终端能够直接找到 Python 安装目录里的脚本，也可以把第二行简写为 `scalarpulse demo --open`。使用 `python -m scalarpulse` 在 Windows 和虚拟环境中通常更稳妥。

## 接入一个训练循环

下面的示例可以直接运行，不需要先准备模型或数据集：

```python
import math
import time

import scalarpulse

run = scalarpulse.init(
    project="hello-scalarpulse",
    name="first-run",
    config={"learning_rate": 3e-4},
    open_browser=True,
)

for step in range(100):
    run.log(
        {
            "train": {"loss": math.exp(-step / 20)},
            "latency_ms": 20 + step % 7,
        },
        step=step,
    )
    time.sleep(0.05)

run.finish()
input("训练完成，按 Enter 退出……")
```

嵌套字典会展开为 `train/loss` 这样的指标路径。Python 数值、NumPy 标量和只含一个值的 PyTorch Tensor 都可以记录；非标量或 `NaN`、`Inf` 会在写入前被拒绝。

> `run.finish()` 只把本次 run 标记为完成，不会主动关闭看板。进程退出时，本次内嵌看板也会随之结束。

## 哪些模型和算法可以用

只要代码能拿到一个标量，ScalarPulse 就能记录它。

| 场景 | 推荐接入方式 | 常见指标 |
| --- | --- | --- |
| **PyTorch / NLP / 时序网络** | `TorchLogger` 或训练循环里的 `run.log` | loss、F1、学习率、梯度范数 |
| **大模型与 Agent 评估** | 每个样本、任务或评测轮次后调用 `run.log` | 成功率、奖励、延迟、Token 用量 |
| **NumPy / 自定义循环** | 在迭代内部调用 `run.log` | 目标函数、误差、自定义计数 |
| **scikit-learn / 传统机器学习** | 在 fold、stage、`partial_fit` 或训练结束后记录 | MAE、R²、AUC、轮廓系数 |
| **XGBoost 等迭代算法** | 从 callback 或 eval history 中逐步记录 | train/valid loss、AUC |

scikit-learn 和 XGBoost 目前没有内置适配器，需要手动调用 `run.log`。如果一个 `fit()` 把训练过程完全封装起来，ScalarPulse 也无法凭空读取其中的每一步；这时可以记录交叉验证轮次、阶段结果或最终指标。

### 大模型 / Agent 评估示意

```python
import time

for step, case in enumerate(eval_cases):
    started = time.perf_counter()
    result = agent.run(case)

    run.log(
        {
            "agent": {
                "success": float(result.success),
                "reward": result.reward,
                "latency_ms": (time.perf_counter() - started) * 1_000,
                "tokens": result.total_tokens,
            }
        },
        step=step,
    )
```

这段代码展示的是接入位置；`eval_cases`、`agent` 和返回字段请替换为你的评测框架。

## PyTorch 辅助接口

`TorchLogger` 可以顺手记录 Tensor loss、optimizer 学习率和全局梯度范数：

```python
import scalarpulse
from scalarpulse.torch import TorchLogger

run = scalarpulse.init(
    project="text-classification",
    framework="pytorch",
    open_browser=True,
)
logger = TorchLogger(
    run,
    model=model,
    optimizer=optimizer,
    track_grad_norm=True,
)

for step, (inputs, targets) in enumerate(loader):
    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()

    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    # 在 zero_grad() 之前记录，当前 step 的梯度才仍然可用。
    logger.log(loss=loss, metrics={"accuracy": accuracy}, step=step)
    optimizer.step()

run.finish()
```

`scalarpulse.torch` 使用鸭子类型，不会直接导入 PyTorch，因此 PyTorch 仍然是可选依赖。

## 长时间任务

训练时间较长时，建议把看板放在单独的进程中：

```bash
scalarpulse serve --log-dir ./runs --open
```

训练脚本只负责向同一个目录写入：

```python
run = scalarpulse.init(
    project="llm-evaluation",
    log_dir="./runs",
    launch=False,
)
```

这样即使训练进程退出或失败，看板进程仍可查看此前已经写入的数据。它不会自动恢复一个中断的训练任务。

## 看板能力

- 通过 SSE 增量接收实时指标。
- 在历史 run 之间切换；当前一次显示一个 run，不做多 run 叠加比较。
- `1–50` 档移动平滑、暂停/恢复刷新、最近 `240` 点自动跟随和悬停读数。
- 暂停期间最多缓存 `5,000` 个事件；前端每个指标最多保留 `20,000` 个点。
- 明亮/深色主题，导出当前 run 的 CSV。
- 显示运行状态、最新 step、指标数、持续时间和更新时间。

<details>
<summary><strong>Python API</strong></summary>

```python
run = scalarpulse.init(
    project="default",
    name=None,
    config=None,
    tags=None,
    framework=None,
    log_dir=".scalarpulse",
    run_id=None,
    launch=True,
    host="127.0.0.1",
    port=8765,
    open_browser=False,
    quiet=False,
)

run.log({"loss": 0.42}, step=10)
run.finish()                 # 正常完成
# 异常或中断时，改用 run.finish("failed") 或 run.finish("stopped")
run.stop_dashboard()
```

省略 `step` 时会从 `0` 开始自动递增。也可以把 run 当作上下文管理器使用；代码块抛出异常时会自动标记为 `failed`。

</details>

<details>
<summary><strong>命令行与数据格式</strong></summary>

```text
scalarpulse serve [--log-dir DIR] [--host HOST] [--port PORT] [--open]
scalarpulse demo  [--steps N] [--interval SECONDS] [同样的服务参数]
```

每个 run 都有独立目录：

```text
.scalarpulse/
└── 20260824-161500-a1b2c3/
    ├── meta.json
    └── metrics.jsonl
```

每次 `log()` 写入一行 JSONL。格式便于追加、人工阅读，也容易交给 pandas 处理；读取时会忽略意外损坏的最后一行。指标 API 默认最多返回 `20,000` 条记录，单次请求上限为 `100,000` 条。

</details>

## 安全与数据

ScalarPulse 默认只监听 `127.0.0.1`，并且**没有身份验证**。不要把服务直接暴露到不可信网络；如需绑定 `0.0.0.0`，请仅在可信网络中使用，或放在带身份验证的反向代理之后。

`config` 会被保存在本地，并可由看板 API 返回。不要在其中写入密码、Token、API Key 或其他敏感数据。更多说明见 [SECURITY.md](https://github.com/lizhengzhong20-sketch/scalarpulse/blob/main/SECURITY.md)。

## 项目边界

ScalarPulse 当前处于 **Alpha** 阶段，专注于单机、本地、标量指标观察。暂不提供：

- 云同步与多人实验平台；
- 分布式或多进程指标聚合；
- 多 run 曲线叠加比较；
- 图片、直方图、模型产物和 artifact 管理；
- 身份验证与大规模日志索引。

它不是训练器，也不是 PyTorch、TensorBoard 或托管实验平台的替代品。当你需要分布式追踪、团队协作或复杂产物管理时，后者通常更合适。

## 开发与贡献

```bash
git clone https://github.com/lizhengzhong20-sketch/scalarpulse.git
cd scalarpulse
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

欢迎提交 Issue 或功能建议，也可以阅读 [CONTRIBUTING.md](https://github.com/lizhengzhong20-sketch/scalarpulse/blob/main/CONTRIBUTING.md) 参与开发。

## License

[MIT](https://github.com/lizhengzhong20-sketch/scalarpulse/blob/main/LICENSE)

---

<p align="center">
  ScalarPulse 不会让模型收敛得更快，但至少能让你更早知道它有没有在认真学习。
</p>

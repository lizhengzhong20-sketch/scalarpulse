# ScalarPulse Reliability Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复审查报告中的 9 个可靠性问题，并让发布流程受测试和安装验证约束。

**Architecture:** 保留标准库 Python 后端、JSONL 存储和现有单页面外观。记录端拒绝冲突并在成功写入后提交状态；传输端显式通知数据缺口；前端按请求边界与事件序号协调历史和实时更新。历史采样改为固定文件边界的多遍流式读取。

**Tech Stack:** Python >=3.10、pytest、浏览器 JavaScript、Node 内置 test/vm、GitHub Actions。

**Spec:** `C:/Users/李/Documents/Codex/2026-08-24/wo-m/work/scalarpulse-review-2026-09-20.md`（已向用户报告并获准修复）。

## Global Constraints

- Python `>=3.10`；运行时 `dependencies = []`，不得为了测试向用户引入 Node 运行时依赖。
- 保持现有 Tracker 使用方式、中文界面和本地运行方式。
- 不在本次加入 Transformers 回调、跨实验叠加、鉴权等新功能。
- 不删除真实训练日志。根据后续用户授权，修复分支准备 `0.1.1` 补丁版；发布 PyPI 是独立步骤。
- 当前仓库是普通 checkout，分支 `chore/scalarpulse-cleanup`；实施前按用户选择原地修改或创建隔离工作区。
- 新增测试必须先验证在当前实现上失败，再逐项修复；每组修复后运行完整 Python 与已添加的 JS 测试。

## Review Focus

1. 冲突出现在不同字典顺序、空白规范化和多层嵌套时，都应拒绝整次日志而非部分写入。
2. 同一步存在不同 seq、跨指标交错记录、旧格式无 seq 时，不允许新值被旧历史覆盖。
3. 暂停期间收到 reset、恢复请求失败、恢复中又产生缺口时，不能静默宣称已补齐。
4. 文件在扫描/采样期间继续追加或最后一行尚不完整时，不能重复回放或返回半条记录。
5. wheel 脱离源码目录安装后，模块入口、命令行和静态页面仍可用。

## Task 1: 记录端一致性（报告 1、8）

**Files:** `_util.py`、`tracker.py`（均在 `src/scalarpulse/`）；`tests/test_tracker_store.py`。

**Interfaces:** `flatten_metrics(metrics, prefix='') -> dict` 遇到展开后的重名抛 ValueError；`Tracker.log` 签名不变；append 失败不提交 step、seq、summary。

- [ ] 添加并运行失败测试：

```python
@pytest.mark.parametrize('metrics', [
    {'train/loss': 1, 'train': {'loss': 2}},
    {'train': {'loss': 2}, 'train/loss': 1},
    {' loss ': 1, 'loss': 2},
])
def test_metric_collision_rejects_whole_log(tmp_path, metrics):
    with Tracker(log_dir=tmp_path, launch=False, quiet=True) as run:
        with pytest.raises(ValueError, match='duplicate'):
            run.log(metrics)
        assert run.store.records(run.id) == []

def test_failed_append_does_not_advance_state(tmp_path, monkeypatch):
    run = Tracker(log_dir=tmp_path, launch=False, quiet=True)
    run.log({'loss': 1})
    append = run.store.append
    def fail(*args):
        raise OSError('injected append failure')
    monkeypatch.setattr(run.store, 'append', fail)
    with pytest.raises(OSError):
        run.log({'loss': 0.2})
    monkeypatch.setattr(run.store, 'append', append)
    run.log({'accuracy': 0.9})
    run.finish()
    records = run.store.records(run.id)
    assert [(r['seq'], r['step']) for r in records] == [(0, 0), (1, 1)]
    assert run.store.read_run(run.id)['summary'] == {'loss': 1, 'accuracy': 0.9}
```

Run: `python -m pytest tests/test_tracker_store.py -q`。Expected: 新测试失败（未抛异常/序号跳过）。

- [ ] 在 flatten 每次合并子项及叶子前检查重名；计算 record 不推进状态，append 返回后再提交。

```python
items = flatten_metrics(value, full_name) if isinstance(value, Mapping) else {
    full_name: to_scalar(value, name=full_name)
}
for key, scalar in items.items():
    if key in flattened:
        raise ValueError(f'duplicate metric name: {key}')
    flattened[key] = scalar

# Tracker.log: execute only after constructing record successfully.
self.store.append(self.id, record)
self._step = max(self._step, actual_step + 1)
self._seq += 1
self._summary.update(flattened)
```

- [ ] Run: `python -m pytest -q`。Expected: 全绿，包括现有线程安全与显式 step 测试。
- [ ] 记录 diff 与测试结果，保留为本地改动，待整体审查后统一决定提交。

## Task 2: 有界历史读取（报告 7）

**Files:** `src/scalarpulse/store.py`；`tests/test_tracker_store.py`。

**Interfaces:** 保持 `records(run_id)` 的完整读取语义；`state(..., max_records=N)` 在 N>0 时走独立有界采样路径。内存 O(N + 指标种类数)，不随总日志条数增长。N<=0 保持原来完整读取语义。

- [ ] 新增长日志内存测试，在生成文件之后启动 tracemalloc；10 万条、max_records=10 时峰值应低于 16 MiB，并保留首尾。补充 limit=1、空日志、损坏行、稀疏及交错指标案例。

```python
tracemalloc.start()
try:
    state = store.state('run', max_records=10)
    _, peak = tracemalloc.get_traced_memory()
finally:
    tracemalloc.stop()
assert len(state['records']) == 10
assert state['records'][0]['step'] == 0
assert state['records'][-1]['step'] == 99999
assert peak < 16 * 1024 * 1024
```

Run: `python -m pytest tests/test_tracker_store.py -q`。Expected: 内存测试失败；现有采样端点测试保持绿。

- [ ] 打开一个二进制文件句柄并固定 EOF；所有遍历仅处理这个边界以内以换行结束的 JSON 字典。第一遍统计有效记录数及每个指标出现次数；按现有排序/配额规则计算各指标的均匀出现序号。

```python
def pick_positions(count, quota):
    if quota <= 0 or count == 0:
        return set()
    if count <= quota:
        return set(range(count))
    if quota == 1:
        return {count - 1}
    return {round(i * (count - 1) / (quota - 1)) for i in range(quota)}
```

第二遍以指标出现计数选取至多 N 个记录索引；重叠产生剩余预算时，根据未选记录总数计算全局均匀位置。第三遍收集已选及补位记录，保持文件顺序。每遍用同一个 EOF，追加内容只在下次请求出现；不构造全量 record/index 数组。损坏 JSON 保持跳过策略。

- [ ] Run: `python -m pytest -q`。Expected: 全绿。重复运行审查的 10 万条内存探针并报告实测值，而非预估。
- [ ] 记录内存和 IO 权衡：仍为 O(日志长度) 扫描，解决内存膨胀，不宣称已经实现索引或常数时间查询。

## Task 3: 实时传输恢复（报告 5、6）

**Files:** `src/scalarpulse/server.py`；`tests/test_server.py`。

**Interfaces:** 保留 metric/run SSE；增加 `reset` 事件，payload 为 `{"reason":"overflow"}`。前端 Task 4 收到后重新拉取持久化历史。reset 后的点不能取代恢复历史。

- [ ] 添加 watcher 追加竞争测试：在第一次读取尾部后追加 seq2，合并后续扫描事件应恰好为 `[1,2]`，再扫描应为空。添加 overflow 测试，慢订阅者必须收到 reset，最后点为最新 seq；快速订阅者不应被强制 reset。

```python
broker = EventBroker()
events = broker.subscribe()
for seq in range(1100):
    broker.publish('metric', {'seq': seq})
received = []
while not events.empty():
    received.append(events.get_nowait())
assert any(kind == 'reset' for kind, _ in received)
assert received[-1] == ('metric', {'seq': 1099})
```

Run: `python -m pytest tests/test_server.py -q`。Expected: reset 缺失和重复序列断言失败。

- [ ] watcher 的 `stream.read()` 改成 `stream.read(size - position)`，令读取位置和尾部指纹使用同一边界。
- [ ] overflow 时在 broker 锁内清空该订阅者积压、放入 reset，再放当前事件；保留 unsubscribe 行为。更新旧测试中“默默丢最老事件”的期望为显式失效协议。

```python
except queue.Full:
    while True:
        try:
            events.get_nowait()
        except queue.Empty:
            break
    events.put_nowait(('reset', {'reason': 'overflow'}))
    events.put_nowait((event, data))
```

- [ ] Run: `python -m pytest -q`。Expected: 全绿，包括半行续写、截断、SSE 断开回收测试。

## Task 4: 前端状态一致性（报告 2、3、4、9，并消费 reset）

**Files:** `src/scalarpulse/static/index.html`；新增 `tests/dashboard.test.cjs`，只使用 Node 内置模块。

**Interfaces:** 对外 API 不变；点保留 seq；每次 loadSnapshot 有递增请求编号；过期请求不能改变选择或错误状态；完成事件使用 ended_at。reset 标记需恢复，暂停期间延后恢复且不丢失这个标记。

- [ ] 提取生产函数到 Node vm 测试上下文，render/DOM 最小桩，实际调用 normalizedRun、applyPayload、ingestSnapshot 和 loadSnapshot。以审查脚本作为 fixture，新增以下断言：

```javascript
assert.deepEqual([...normalizedRun({id:'a', records:[{
  step:1, metrics:{epoch:3,time:4,x:5,loss:6}
}]}).metrics.keys()], ['epoch','time','x','loss']);
// Complete event then late running snapshot must retain completed.
assert.equal(app.runs.get('a').status, 'completed');
// seq 1 live loss=9 then seq 0 snapshot loss=1 at the same step.
assert.equal(app.runs.get('a').metrics.get('loss')[0].value, 9);
// Resolve real loadSnapshot(B) before loadSnapshot(A).
assert.equal(app.selectedRunId, 'b');
// started 00:00, last metric 00:00:01, ended 00:10.
assert.equal(app.runs.get('a').updatedAt, '2026-01-01T00:10:00.000Z');
```

另测 reset 后真实 loadSnapshot 被调度并合入缺失历史、暂停期间 reset 恢复后补数、fetch 失败不清除恢复需求、缓冲再次溢出不会显示“已补齐”。

Run: `node --test tests/dashboard.test.cjs`。Expected: 新增用例因当前行为失败，不得因为测试桩缺属性而失败。

- [ ] 仅对扁平记录过滤元数据字段。历史点与实时点保留 `seq`，同 step 优先较大 seq；缺少 seq 时比较有效时间，完全相同再按事件边界规则确定优先级。

```javascript
const nested = record.metrics && typeof record.metrics === 'object';
const values = nested ? record.metrics : record;
// Apply reserved-field filtering only when !nested.
const seq = toFinite(record.seq);
run.metrics.get(name).push({step, value:numeric, time:normalizedTime, seq});
```

- [ ] 使用请求编号丢弃过期成功/失败响应，记录请求开始时的实时变更代次；快照只覆盖在该请求开始后未被实时元数据更新的字段。点的 seq 比较仍必须生效，不能只靠“最后一次请求”。

```javascript
const requestId = ++app.snapshotRequestId;
// after both awaited fetch and json, before changing any app/DOM state:
if (requestId !== app.snapshotRequestId) return;
```

- [ ] 完成事件保留 ended_at，晚到的指标不得把结束时间回退。reset 不当作普通 run payload；合并补数请求，失败保留 dirty 标记并重试，暂停时仅标记恢复需求，继续后重新同步。
- [ ] Run: `node --test tests/dashboard.test.cjs` 和 `python -m pytest -q`。Expected: 全绿。启动本地服务进行选择、刷新、暂停、结束后的页面冒烟验证。

## Task 5: 测试与发布门禁

**Files:** `.github/workflows/tests.yml`、`.github/workflows/publish.yml`、`README.md`、`CHANGELOG.md`（若已有）。

**Interfaces:** release 发布必须依赖同一工作流中的测试、构建、wheel 安装冒烟成功；不能依靠另一工作流恰好成功。

- [ ] CI 增加 Node 环境和 `node --test tests/dashboard.test.cjs`；Python 矩阵保留 Windows/Linux 和 3.10–3.13。
- [ ] 发布 build 作业在构建前运行 Python/JS 测试，构建后运行 `python -m twine check dist/*`，在新建环境且源码目录之外验证 wheel 的 module、CLI、静态页面资源；publish 继续 `needs: build`，因测试已进入 build 的硬依赖链而受门禁控制。

```yaml
- name: Test Python and dashboard
  run: |
    python -m pytest -q
    node --test tests/dashboard.test.cjs
- name: Validate distributions
  run: python -m twine check dist/*
```

- [ ] 记录中文变更说明：冲突会抛异常；缺口自动同步；返回历史仍会采样；内存改进不代表完整 CSV 导出或索引已经实现。
- [ ] 本地运行上述测试、构建、wheel 冒烟命令；若依赖下载受限，明确记录未验证部分，不触发实际发布来“验证”。
- [ ] 独立整体代码审查，修复重要新增发现；运行 `git diff --check` 和完整测试；交付变更摘要、测试结果及未覆盖的真实 GPU 场景。不自动推送或发布。

## Handoff

推荐原生顺序执行：同一实施者逐项修改，最后一次独立审查。原因是后端 reset 与前端恢复、采样边界与前端快照有共同接口，集中实施更易保持一致；不需要每项都启动新实施代理。

用户已确认在当前目录顺序实施。2026-09-20 修复验证：45 项 Python、14 项 JS 测试通过；独立审查的暂停切换问题已用失败→通过回归测试修复。后续决定建立基于最新 `origin/main` 的 PR 修复分支，准备 `0.1.1`。详细验证与保留限制见工作区 work/scalarpulse-fix-verification-20260920.md。

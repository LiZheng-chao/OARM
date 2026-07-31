# OARM 场景与闭环实验阶段汇总

日期：2026-07-31  
项目：OARM-Planner on YOPO  
目的：筛选适合 OARM 论文方向的闭环场景，并判断当前 A 系列权重是否足够支撑后续主方法实验。

## 1. 当前方法状态

当前主要测试的方法：

| 方法 | 含义 | 当前定位 |
|---|---|---|
| clean YOPO | YOPO 原策略 exact adapter，只加日志，不改 baseline | baseline |
| A0_yopo_preserve | OARM wrapper，但保持 YOPO 行为 | parity / sanity |
| A1_yopo_preserve | YOPO preserve + margin/risk auxiliary head | 当前已训练权重 |
| A2_margin_a020_b000 | A1 权重 + online margin selector，alpha=0.2, beta=0 | selector ablation |

重要限制：

- 目前权重主要推进到 A1/A2 阶段，尚未进入完整 OARM 主方法。
- 当前 A 系列闭环中基本仍是 `selected_progress_rate=1.0`，没有真正使用 probe / brake / yield 行为覆盖。
- 因此，在复杂遮挡场景中不能期待 A1/A2 已经明显胜过 YOPO；当前阶段更适合验证场景、日志、GT 标注和闭环流程。

## 2. 评估标准

近期统一使用：

| 指标 | 标准 |
---|---|
| 成功 | `success_distance = 2.0m` |
| 碰撞 | `collision_clearance = 0.25m` |
| 目标 | `goal = (50, 0, 2)` |
| GT | 与 Simulator 完全匹配的 `pointcloud-0.ply` |

采用 2m 成功阈值的原因：

- 在 goal50 任务中，1m 成功圈过严，clean YOPO 和 A1 都会在终点附近高速掠过后绕圈。
- 用 2m 重算后，之前的“绕圈失败”变成正常到达，说明该问题主要是终端收敛/成功阈值问题，不是 OARM 安全行为问题。

## 3. 场景演进总结

### 3.1 随机墙体场景

| 场景 | 结果 | 判断 |
|---|---|---|
| `oarm_wall_blind_corner_s0`, goal35 | clean/A0/A1/A2 都成功、0 碰撞，但 hidden risk = 0 | 太短、太容易 |
| `oarm_wall_blind_corner_long_s0`, goal50 | hidden risk 有覆盖，A0/A1/A2 约 13%-15%，但所有方法早期碰撞 | 太难，开局障碍/缝隙过强 |
| `oarm_wall_blind_corner_long_light_s0`, goal50 | 不碰撞，但 1m 不成功，后半段太空，hidden risk = 0 | 太容易且不贴合 OARM |

结论：随机墙体不稳定，不适合作为主实验 benchmark。需要确定性场景。

### 3.2 确定性 blind gate 系列

| 场景 | 设计思路 | 结果 | 判断 |
|---|---|---|---|
| `blind_gate_s0` | 初版 deterministic gate，左右可绕，一侧藏风险 | 2m 下 clean/A1 都成功，A1 hidden risk 0.45% | 流程正确，但 risk 太少 |
| `blind_gate_v2_s0` | 交替门，中段遮挡，终点开阔 | 四方法都成功、0 碰撞，A1 hidden risk 0.79% | 稳定，但信号弱 |
| `blind_gate_v2_s1` | v2 镜像 | clean/A1 成功、0 碰撞，hidden risk 0 | 镜像不产生风险 |
| `blind_gate_v3_s0` | 更贴边、更靠近中心风险 | clean/A1 成功、0 碰撞，但 hidden risk 0 | 实际轨迹绕到 y≈-7m，绕开风险 |
| `blind_gate_v4_s0` | 加走廊边界，限制外圈绕行，中段交替遮挡 | clean/A1 成功、0 碰撞，A1 hidden risk 3.63% | 当前最好的候选场景 |

## 4. 关键闭环结果表

### 4.1 `blind_gate_s0_goal50`, success=2m

| 方法 | success | collision | min clearance | path time | hidden risk cov | gt RMVR cov |
|---|---:|---:|---:|---:|---:|---:|
| clean YOPO | 1.0 | 0.0 | 1.073m | 20.36s | 0 | 0 |
| A1 | 1.0 | 0.0 | 1.104m | 20.28s | 0.45% | 0.45% |

结论：2m 阈值解决终点绕圈问题，但 hidden risk 太少。

### 4.2 `blind_gate_v2_s0_goal50`, success=2m

| 方法 | success | collision | min clearance | path time | hidden risk cov | gt RMVR cov |
|---|---:|---:|---:|---:|---:|---:|
| clean YOPO | 1.0 | 0.0 | 1.681m | 15.26s | 0 | 0 |
| A0 | 1.0 | 0.0 | 1.689m | 15.19s | 0.80% | 0.80% |
| A1 | 1.0 | 0.0 | 1.690m | 15.28s | 0.79% | 0.79% |
| A2 | 1.0 | 0.0 | 1.693m | 15.12s | 0 | 0.40% |

结论：四方法稳定，A2 没副作用，但 OARM 信号弱。

### 4.3 `blind_gate_v2_s1_goal50`, success=2m

| 方法 | success | collision | min clearance | path time | hidden risk cov |
|---|---:|---:|---:|---:|---:|
| clean YOPO | 1.0 | 0.0 | 1.658m | 15.21s | 0 |
| A1 | 1.0 | 0.0 | 1.661m | 15.15s | 0 |

结论：镜像稳定但无 hidden risk。

### 4.4 `blind_gate_v3_s0_goal50`, success=2m

| 方法 | success | collision | min clearance | path time | hidden risk cov |
|---|---:|---:|---:|---:|---:|
| clean YOPO | 1.0 | 0.0 | 1.695m | 15.42s | 0 |
| A1 | 1.0 | 0.0 | 1.689m | 15.49s | 0 |

补充观察：

- 实际路径统计显示，无人机在中段绕到下侧，约 `y=-7m`。
- 原本中心附近设置的风险点没有被实际轨迹贴近，所以 GT hidden risk 为 0。

### 4.5 `blind_gate_v4_s0_goal50`, success=2m

| 方法 | success | collision | min clearance | path time | hidden risk cov | gt RMVR cov | selected RMVR GT |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean YOPO | 1.0 | 0.0 | 0.964m | 20.95s | 0 | 0 | null |
| A1 | 1.0 | 0.0 | 0.968m | 20.85s | 3.63% | 3.63% | 1.0 |

v4 关键指标：

- A1 `hidden_risk_gt_coverage = 0.0363`
- A1 `gt_rmvr_coverage = 0.0363`
- A1 `selected_rmvr_gt = 1.0`
- A1 `reaction_margin_violation_rate = 1.0`
- A1 `selected_traj_collision_gt = 0.0`
- clean/A1 都成功、0 碰撞

结论：v4 是当前最好的场景。它既能跑通，又出现了明确的 hidden-risk / RMVR 信号。

## 5. 当前总体判断

### 5.1 场景方面

当前最有价值的场景是：

```text
oarm_blind_gate_v4_s0_goal50
```

原因：

- clean YOPO 和 A1 都可以完成任务，没有碰撞。
- 不再出现 goal50 终点绕圈污染，因为采用 2m 成功阈值。
- hidden risk 覆盖从 v2 的约 0.8% 提升到 v4 的 3.63%。
- RMVR GT 也同步提升到 3.63%。
- 场景结构可解释：走廊边界限制外圈绕行，中段交替门和中心短柱制造 near-route hidden risk。

不足：

- hidden risk 覆盖仍未达到理想的 5%-10%。
- 当前只有 s0，还需要镜像 s1 或更多 seed/variant 验证。
- clean YOPO 没有 OARM candidate GT 字段，所以 hidden risk 表主要来自 A0/A1/A2 类型日志；clean 主要比较执行指标。

### 5.2 算法方面

当前 A1/A2 不应被期待在难场景中明显胜过 YOPO：

- A1 只是 YOPO preserve + auxiliary margin/risk head。
- A2 只是简单 online selector，`alpha=0.2, beta=0`。
- 当前闭环仍然几乎全是 `progress` candidate，没有真正体现 probe/brake/yield 这些 OARM 行为。
- 因此，继续只调场景的边际收益会下降。

我认同用户的判断：**应该尽快推进到主方法训练/部署，而不是长期停留在 A 阶段。**

## 6. 推荐下一步

### P0：补齐 v4_s0 四方法

先在当前最好的 v4_s0 上补：

```text
clean YOPO
A0_yopo_preserve
A1_yopo_preserve
A2_margin_a020_b000
```

目的：

- 验证 A2 在 v4 hidden-risk 场景下是否有副作用或收益。
- 确认 A0/A1/A2 的 GT RMVR coverage 是否一致。

### P1：跑 v4_s1 镜像

如果 v4_s1 也能保持：

```text
success = 1
collision = 0
hidden_risk_gt_coverage > 0
```

那么 v4 系列可以作为正式场景候选。

### P2：推进主方法

建议并行推进主方法，而不是继续只调场景：

- 训练/部署完整 OARM candidate 行为。
- 让 probe/brake/yield 或 typed candidates 真正进入闭环选择。
- 再用 v4 场景检验主方法是否能在 hidden risk / RMVR 上相对 A0/A1/A2 有优势。

## 7. 给 GPT 的分析问题

请重点分析：

1. 当前 v4_s0 是否已经足够作为 OARM 主实验候选场景？
2. hidden risk coverage 3.63% 是否够用，还是需要继续把场景调到 5%-10%？
3. 在当前 A1/A2 仍然以 YOPO preserve/progress candidate 为主的情况下，是否应该停止场景微调，优先推进完整 OARM 主方法训练？
4. 对论文叙事而言，v4_s0 的结果更适合放在 sanity / scenario validation，还是可以作为正式闭环主结果的一部分？
5. 如果继续改场景，应该增加遮挡风险密度，还是保持 v4 并扩展多个 seed/variant？


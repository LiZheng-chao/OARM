# OARM v4 场景与当前算法阶段分析

日期：2026-07-31  
项目：OARM-Planner on YOPO  
用途：给 GPT/导师/自己复盘当前场景筛选结果，并判断是否应该继续改场景，还是推进完整 OARM 主方法。

## 1. 背景

当前 OARM 代码仍处在 A 阶段：

| 方法 | 说明 | 当前角色 |
|---|---|---|
| clean YOPO | YOPO 原策略 exact adapter，只加日志，不改 YOPO baseline | baseline |
| A0_yopo_preserve | OARM wrapper，但保持 YOPO 行为 | parity / sanity |
| A1_yopo_preserve | YOPO preserve + margin/risk auxiliary head | 当前主要训练权重 |
| A2_margin_a020_b000 | A1 权重 + online margin selector，alpha=0.2, beta=0 | selector ablation |

重要事实：

- A1/A2 还不是完整 OARM 主方法。
- 当前闭环里基本仍是 `selected_progress_rate=1.0`，没有真正用到 probe / brake / yield 等 OARM 行为候选。
- 所以目前更适合验证场景、GT 后处理、执行日志、hidden risk / RMVR 指标，而不是期待 A1/A2 在困难场景明显胜过 YOPO。

## 2. 统一评估设置

| 项 | 设置 |
|---|---|
| goal | `(50, 0, 2)` |
| success distance | `2.0m` |
| collision clearance | `0.25m` |
| yaw mode | `goal` |
| GT | 与 Simulator 使用同一份 deterministic PLY |

使用 2m 成功阈值的原因：

- 在 goal50 闭环中，1m 成功圈过严，clean YOPO 和 A1 都曾在终点附近绕圈。
- 2m 重算后，之前的“终点附近绕圈失败”变成正常到达。
- 说明绕圈主要是 YOPO/OARM 当前控制闭环的末端收敛问题，而非安全/避障问题。

## 3. 场景演进概览

| 场景 | 结果 | 判断 |
|---|---|---|
| random wall goal35 | 全部成功，hidden risk=0 | 太短太容易 |
| random wall long goal50 | hidden risk 13%-15%，但全部早期碰撞 | 太难，不可用 |
| random wall light goal50 | 不碰撞，但后半段太空，hidden risk=0 | 不贴合 OARM |
| blind_gate_s0 | 2m 成功，A1 hidden risk 0.45% | 流程正确但信号弱 |
| blind_gate_v2_s0 | 四方法成功，A1 hidden risk 0.79% | 稳定但信号弱 |
| blind_gate_v3_s0 | 成功但 hidden risk=0 | 实际轨迹绕到外侧，避开风险 |
| blind_gate_v4_s0 | 四方法成功、0 碰撞，hidden risk 约 3.5%-3.6% | 当前最好 |
| blind_gate_v4_s1 | 成功、0 碰撞，hidden risk=0 | 镜像不产生有效风险 |

## 4. v4_s0 四方法结果

场景：

```text
oarm_blind_gate_v4_s0_goal50
```

结构：

- 使用 deterministic PLY。
- 加上下走廊边界，避免无人机从外侧绕开所有风险。
- 中段设置交替门和中心短柱。
- 终点附近保持开阔，避免终点绕圈污染。

### 4.1 结果表

| 方法 | success | collision | min clearance exec | path time | hidden risk cov | GT RMVR cov | selected RMVR GT |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean YOPO | 1.0 | 0.0 | 0.964m | 20.95s | 0 | 0 | null |
| A0_yopo_preserve | 1.0 | 0.0 | 0.969m | 20.92s | 3.62% | 3.62% | 1.0 |
| A1_yopo_preserve | 1.0 | 0.0 | 0.968m | 20.85s | 3.63% | 3.63% | 1.0 |
| A2_margin_a020_b000 | 1.0 | 0.0 | 0.973m | 20.19s | 3.45% | 3.45% | 1.0 |

### 4.2 关键观察

正面：

- 四方法全部成功，且无碰撞。
- v4_s0 相比 v2/v3 明显提高 hidden risk 覆盖。
- A0/A1/A2 的 `selected_rmvr_gt = 1.0`，说明这个场景确实出现了 OARM 标注认为有反应裕度风险的片段。
- A2 没有造成安全退化，反而 path time 略短，min clearance 略高。

不足：

- hidden risk coverage 约 3.5%-3.6%，仍未达到理想的 5%-10%。
- clean YOPO 没有 OARM candidate GT 字段，所以 hidden risk/RMVR 主要看 A0/A1/A2。
- A0/A1/A2 的结果非常接近，说明 A1/A2 当前并没有真正带来明显行为改进。

## 5. v4_s1 镜像结果

场景：

```text
oarm_blind_gate_v4_s1_goal50
```

| 方法 | success | collision | min clearance exec | path time | hidden risk cov |
|---|---:|---:|---:|---:|---:|
| clean YOPO | 1.0 | 0.0 | 1.123m | 21.17s | 0 |
| A1_yopo_preserve | 1.0 | 0.0 | 1.115m | 21.14s | 0 |

判断：

- v4_s1 作为镜像场景稳定，但没有产生 hidden risk。
- 说明 scene geometry 和实际执行轨迹之间仍然强耦合；简单镜像不一定有效。
- 如果需要多场景，应该按真实轨迹反向设计，而不是只镜像障碍。

## 6. 关于“遇到障碍物直接抬高飞过去”的观察

用户观察：

> 在 v4 场景下，无人机有时面前有障碍物时直接提高飞行高度，而不是绕开。

判断：

这是一个重要问题。它说明当前 YOPO/A 阶段策略会利用 3D 空间中的高度自由度进行“vertical escape”。如果论文想证明的是遮挡下的反应裕度、绕行决策、probe/brake/yield 行为，那么单纯让无人机抬高飞越障碍可能会削弱实验说服力。

可能原因：

1. YOPO 原策略本身允许 3D 轨迹，遇到障碍时抬升是合理避障方式。
2. 当前 A0/A1/A2 基本保持 YOPO progress 行为，还没有主方法的显式行为候选约束。
3. 场景没有天花板或高度限制，飞升是一条容易且安全的逃逸路径。
4. 当前评估只看 collision/success/clearance/RMVR，没有惩罚过度高度变化。

这不一定是 bug，但对 OARM 论文叙事有风险。

## 7. 是否继续改场景？

我的判断：

不建议继续长期只调场景。v4_s0 已经足够证明：

- deterministic 场景流程能跑通；
- GT 后处理和 execution monitor 能工作；
- hidden risk/RMVR 可以被激活；
- A 阶段方法能稳定闭环。

但 v4_s0 也暴露：

- A1/A2 没有明显优于 A0/clean；
- vertical escape 会削弱“绕行/反应决策”叙事；
- 当前 A 阶段不具备完整 OARM 行为表达。

因此，应该推进完整 OARM 主方法。

## 8. 推荐下一步

### P0：推进主方法训练/部署

目标：

- 不再只停留在 YOPO preserve + auxiliary head。
- 让 OARM typed candidates / probe / brake / yield 等行为真正进入闭环选择。
- 让主方法在 hidden risk/RMVR 片段中表现出可解释差异。

### P1：加入高度约束或高度惩罚

如果论文希望强调绕行而非飞越，可以考虑：

1. 场景层面：加 ceiling 或低高度障碍，让直接飞越不可行。
2. 策略层面：限制 candidate z 范围，例如 `selector-min-traj-z / selector-max-traj-z`。
3. 评估层面：增加 altitude deviation / max z / vertical escape rate 指标。
4. 训练层面：对过度抬升加入代价。

注意：如果加 ceiling，要避免让场景重新变成“全方法早撞”的不可用场景。

### P2：保留 v4_s0 作为 scenario validation

v4_s0 可以作为：

- deterministic occlusion validation；
- hidden risk/RMVR pipeline demonstration；
- A0/A1/A2 sanity check；
- 后续主方法上线后的第一批闭环测试场景。

但暂时不建议把 v4_s0 单独作为最终主实验结论。

## 9. 给 GPT 的问题

请重点分析：

1. v4_s0 的 hidden risk coverage 约 3.6%，是否足够作为 OARM 场景候选？
2. 当前无人机通过抬高飞越障碍完成任务，这会不会削弱 OARM 论文叙事？
3. 应该优先：
   - 继续调场景，提高 hidden risk 到 5%-10%；
   - 加高度约束/天花板；
   - 还是推进完整 OARM 主方法训练？
4. 在 A1/A2 尚未真正使用 probe/brake/yield 的情况下，是否合理期待其在 v4_s0 明显优于 YOPO？
5. 后续主方法应该如何设计闭环指标，才能体现 reaction margin / hidden risk 的贡献，而不是只看 success/collision？


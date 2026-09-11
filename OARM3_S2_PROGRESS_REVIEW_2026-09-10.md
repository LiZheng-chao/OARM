# OARM3-S2 当前实验进展与方向评审

更新时间：2026-09-10

## 1. 研究目标与当前方法

目标是在不修改原始 YOPO baseline 的前提下，为 YOPO 候选轨迹增加遮挡风险与反应窗口建模，并通过 KEEP、RERANK、BRAKE 在线决策降低遮挡场景中的碰撞风险。

当前主线：

1. 保留 YOPO 原始候选生成、backbone、代价输出和 top-1 定义。
2. critic 输入 YOPO 特征、候选 cost 和候选轨迹几何。
3. 使用两阶段概率模型：先预测是否发生有效风险交互，再预测条件于交互的 reaction-window hazard/CDF。
4. 在线根据传感、规划、控制和制动延迟计算真实 reaction budget `tau`。
5. 风险定义为 `P(interaction) * P(reaction_window < tau | interaction)`，而非固定 risk logit。
6. 使用独立 calibration split 校准概率并估计 empirical upper slack。
7. selector 低风险时 KEEP YOPO top-1；有更安全候选时 RERANK；否则执行可行 BRAKE 轨迹。
8. 高度、目标进展、depth clearance、制动动力学和 brake latch 共同限制不合理切换。

原始 YOPO baseline 保持独立，OARM 修改没有写回 baseline 实现。

## 2. 已完成的实现

- CandidateRMCritic 已接入 YOPO-preserve 网络。
- Dataset 已输出 interaction valid、zero-window、positive-window、no-entry、censored 和 reaction-window 标签。
- hazard/CDF loss、validity loss 和 mask 已接入。
- 训练仅允许 rm_critic 更新，YOPO 参数冻结。
- hazard CDF 对 tau 单调，并有 smoke/unit test。
- checkpoint 保存 hazard bin、horizon 和训练阶段元数据。
- ROS 已接入实时 latency、reaction budget、calibration、selector、BRAKE、brake latch 和日志。
- GT 标注、execution monitor 和 scenario benchmark 后处理链路已经可用。

## 3. 正式训练结果

目录：`OARM/runs/oarm3_s2_formal`

- Seed 0：best epoch 11，best validation metric 0.0422107，训练 16 epoch 后正常早停。
- Seed 1：best epoch 28，best validation metric -0.0813892，训练 33 epoch 后正常早停。
- 两个 seed 均无 NaN、Inf、OOM 或 traceback。
- 两者都只有 6 个 rm_critic 参数张量可训练，YOPO backbone/head 冻结。
- hazard horizon 为 1.6666666667 s，与 YOPO 轨迹时域一致。

当前首选 checkpoint：

`OARM/runs/oarm3_s2_formal/seed1/OARM_0/best_val.pth`

## 4. 离线评估

Seed 0 validation（366 样本）：

- validity balanced accuracy/AUROC/AUPRC：0.7593 / 0.8315 / 0.6382
- risk balanced accuracy：0.7536-0.7623
- risk AUROC/AUPRC：0.8276-0.8337 / 0.5459-0.6235
- pairwise ordering：0.7060-0.7449
- oracle rescue：0.2817-0.3375
- monotonicity violation：0

Seed 1 validation（366 样本）：

- validity balanced accuracy/AUROC/AUPRC：0.7776 / 0.8486 / 0.6243
- risk balanced accuracy：0.7436-0.7685
- risk AUROC/AUPRC：0.8348-0.8450 / 0.5240-0.5928
- pairwise ordering：0.7180-0.7437
- oracle rescue：0.2439-0.2658
- monotonicity violation：0

Seed 1 train（3271 样本）的 risk AUROC 为 0.8854-0.8916，pairwise ordering 为 0.7418-0.7634。

判断：critic 已学到有效风险判别和候选排序信号。约 24%-34% 的 oracle rescue 表明部分 YOPO top-1 失败帧中存在更安全候选。离线结果支持受控闭环诊断，但尚不能证明闭环收益。


## 5. Calibration 结果

可用文件：`OARM/runs/oarm3_s2_formal/calibration_s0/risk_calibration.json`

- 场景：`oarm_ceiling_gate_v1_s0`
- calibration episode/seed：70、71
- selector 关闭，仅采集 nominal YOPO 数据
- 使用候选记录：31,470；缺失标签 240/31,710（约 0.76%）
- bias：0.5245497
- temperature：6.9718971
- empirical upper slack：0.0816740

| 指标 | 校准前 | Platt/temperature 后 | 加 upper slack 后 |
|---|---:|---:|---:|
| Brier | 0.37696 | 0.23797 | 0.24464 |
| ECE | 0.34284 | 0.07735 | 0.08163 |
| NLL | 1.07800 | 0.66943 | 0.68548 |

校准明显改善 Brier、ECE 和 NLL，但只有两个高度相似 episode。31,470 是候选记录数，不等于独立样本数。当前 calibration 可用于诊断，不宜直接作为最终论文证据。

`calibration_s1/` 是不完整重复/残留采集，缺少 GT 和 calibration 文件，不能视为独立 calibration set。

## 6. 关键阈值问题

当前 calibration 加 empirical upper slack 后：

| 分位数 | Risk upper bound |
|---|---:|
| minimum | 0.4733 |
| 5% | 0.5702 |
| 25% | 0.6383 |
| median | 0.6589 |
| 75% | 0.7098 |
| 95% | 0.7766 |
| maximum | 0.8395 |

| 阈值 | 候选通过比例 | top-1 可 KEEP 的帧 | 至少有一个安全候选的帧 |
|---:|---:|---:|---:|
| 0.45 | 0.00% | 0.00% | 0.00% |
| 0.55 | 3.63% | 4.54% | 8.75% |
| 0.60 | 6.45% | 6.24% | 13.72% |
| 0.65 | 40.66% | 36.00% | 66.93% |
| 0.70 | 71.55% | 60.97% | 91.72% |
| 0.75 | 88.90% | 83.25% | 99.20% |

旧 locked protocol 使用 `keep=0.45`、`safe=0.55`。新 calibration 下 KEEP 永远不能触发，约 91% 帧没有满足 safe 阈值的普通候选，selector 会被系统性推向 RERANK/BRAKE。这可能解释此前频繁制动、锁存、震荡和无法接近目标。

此前给出的长闭环命令只是参数完全显式的诊断版本，不代表这些参数已经成为正式论文协议。正式协议冻结后应整理成短脚本或配置入口。

## 7. 当前闭环状态

`oarm3_s2_formal` 下尚未发现 selector-enabled 且使用当前 calibration 的正式 intervention 闭环日志。

已有三个 exec 日志都是 `oarm_ceiling_gate_v1_s0` 的 selector-disabled calibration 采集，均到达目标且无碰撞。这验证了 nominal 执行和采集链路，但没有验证 KEEP/RERANK/BRAKE 的闭环收益。

## 8. 建议方向

目前不建议重新训练：

1. 保留 seed1 `best_val.pth`。
2. 暂停 held-out `oarm_ceiling_gate_v1_s1`，不得用它调阈值。
3. 先在 validation 场景 `oarm_blind_gate_v4_s0` 做一次 seed77 selector 代码回归。
4. `keep=0.65`、`safe=0.70` 只能作为 calibration coverage 推出的诊断起点，不是最终论文阈值。
5. 检查 KEEP/RERANK/BRAKE 比例、risk before/after、collision、success、timeout、stale depth、goal progress、brake latch 和轨迹震荡。
6. 单次闭环稳定后，增加相互独立的 calibration 场景和 episode。
7. 仅使用 calibration/validation 数据确定阈值，再冻结 checkpoint、calibrator、selector 和控制设置。
8. 最后运行 held-out s1、多 seed 对照和消融。

## 9. 希望 GPT 重点评审

1. 两阶段 `P(interaction) * P(reaction_window < tau | interaction)` 是否适合当前标签分布？
2. empirical upper slack 应直接用于在线 selector，还是只作为报告上界，在线使用 calibrated point probability？
3. keep/safe 应按 calibration coverage 选择，还是在 calibration/validation 上优化安全-效率目标？
4. temperature 约 6.97 是否意味着 logits 过于极端，或 calibration 与离线验证存在分布差异？
5. 两个高度相关 episode 是否不足以支持论文级 calibration，至少应增加多少独立场景和 episode？
6. 是否应将 point-calibrated selector 与 upper-bound selector 作为消融？
7. 当前 train/validation 差距是否可接受，是否需要第三个训练 seed？
8. KEEP/RERANK/BRAKE 是否继续作为主线，还是先简化为 KEEP/RERANK，把 BRAKE 作为独立安全层？

## 10. 阶段判断

- 数据标签与两阶段 risk critic：已完成
- 正式 S2 critic 训练：已完成
- 离线风险、排序和单调性验证：已完成并基本通过
- 初步 calibration：已完成，可诊断但多样性不足
- 旧 selector 阈值兼容性：未通过覆盖率检查
- seed77 intervention 回归：尚未完成
- held-out 闭环测试：尚未开始
- 多 seed 正式对照与消融：尚未开始

当前最需要确认的不是是否重新训练，而是 calibration 上界如何参与在线决策，以及如何在不使用 held-out test 的情况下确定 selector 阈值。


# DeltaLM 扩展实测补充（2026-11-14）

**作者：Carlo Young。原十项 Showcase 与 delta_core.py 均保持不变。**

## 真正训练了什么？

新增的 `core_calibration.py` 用已知的合成教师轨迹，训练 DeltaLM 现有 `Substrate.activity_retention` 与 `Substrate.contact_retention` 两个系数。训练采用差分进化与可选局部优化。参数直接影响实际核心后续演化，**但此任务是小规模合成参数辨识，并不是 DeltaLM 自动发明了新学习规律、跨模态语义理解或 GPT 权重联合训练。**

教师合成系数分别为 0.943 和 0.963；初始系数为 0.971 和 0.986。4 个世界用于训练、3 个完整不同初始种子的世界用于检验。测量 5、10、18、28 步的六段原始关系表达式与绝对接合均值。

| 指标 | 初始参数 | 训练后的参数 |
|---|---:|---:|
| 训练世界平均绝对误差 | 0.09217586 | 约 0.000000013 |
| 未见世界平均绝对误差 | 0.08705747 | 约 0.000000012 |

虽然外层差分进化达到预设迭代上限，局部优化仍将两系数拟合得非常接近合成教师；不能把优化器的 `success=false` 隐藏掉。

## 同一初始世界，结构如何改变？

8 个研究者命名节点，12 条初始关系，固定种子 10，320 步及四次完全相同的外部扰动：

| 条件 | 最终关系数 | 新生关系数 | 最终总活动 |
|---|---:|---:|---:|
| C 原始 DeltaLM 核心 | 44 | 32 | 0.54054831 |
| D 使用校准系数的核心 | 30 | 18 | 0.53968697 |

数量减少不意味着更差，也不代表更好：只有变化本身和对特定训练目标的误差获得了证据。

## 四媒介 A/B/C/D 协议与实际状态

新增 `ABCD_protocol.py`，对文字、图像、音频、视频分别建立 A 原始输入、B 自由延伸、C 真实历史条件、D 训练后核心条件，并将每份条件中模型与外部创作的边界写明。

**现有 GPT 聊天文本示例与已有程序化 PNG/WAV/MP4 仍然有效。** 本补充没有在 OpenAI API 上完成四媒介全组 A/B/C/D：当前执行环境没有 `OPENAI_API_KEY`；`--execute` 必须明确授权并由执行者自行承担调用费用。若缺少凭证，不得伪称已产生 API 结果。

另需特别说明：**OpenAI Videos API / Sora 2 已于 2026-09-24 停止服务**，因此只能提供已有程序化视频或另行集成可用视频服务，不能提交声称来自已关闭 API 的虚假结果。

GPT 图像艺术化样本与真实结构拓扑的可复现可视化不同，不应混用作因果证据。

## 复现

```bash
python -m pip install numpy scipy scikit-learn Pillow
python integrations/open_generation/core_calibration.py --maxiter 6
python integrations/open_generation/ABCD_protocol.py --medium text --branch C
python -m unittest discover -s integrations/open_generation -p test_calibration_extension.py -v
```

全部外部训练集、测量标准及合成教师均公开。历史保存在当前结构中，外部实验日志不构成内核必须检索的历史档案。

# debate_rl

面向多智能体协作与对抗式辩论研究的强化学习-大语言模型融合仓库。

这个仓库当前同时包含两条代码线：

- `debate_rl/`：早期版本，脚本更完整，适合直接跑训练、评估和可视化流程。
- `debate_rl_v2/`：当前研究主线，强调“RL 策略控制 + LLM 文本生成 + 合规反馈 + 在线积累”的统一框架，也是本文档重点说明的对象。

如果你是第一次阅读本仓库，建议把它理解为一个研究型代码库，而不是一个单入口产品。它试图解决的问题是：

**如何让强化学习负责协作策略控制，让大语言模型负责自然语言生成，再通过合规验证、奖励反馈和经验沉淀形成闭环，从而实现可控、可优化、可积累的多智能体协作系统。**

---

## 1. 仓库定位

本仓库最早来源于多角色辩论/会诊场景，后续逐步抽象为面向通用协作任务的框架。当前可以把它分成三层理解：

1. **数值策略层**：用 PPO、MADDPG 等多智能体强化学习方法学习“何时更强硬、何时更保守、何时提高对证据的要求”等高层策略。
2. **语言执行层**：让 LLM 依据角色身份、规则、上下文和 RL 传来的策略信号完成自然语言生成。
3. **闭环改进层**：对生成结果做合规校验、奖励计算、知识蒸馏、技能持久化、提示词演化和在线参数更新。

因此，这个项目不是“RL 直接生成文本”，而是“RL 控制语言协作过程”。

---

## 2. 系统原理

### 2.1 核心思想

在传统多 Agent LLM 系统中，角色行为往往被固定 prompt 决定，系统难以持续优化。本仓库将行为控制拆成两个层次：

- **RL 学策略**：学习连续动作，表达协作风格、对抗强度、证据要求、仲裁偏置等控制信号。
- **LLM 学表达**：根据这些控制信号生成自然语言内容。

这样做有三个直接好处：

- 控制空间更小，更适合强化学习优化。
- 生成空间仍由 LLM 负责，保留语言能力和泛化能力。
- 系统可以显式判断“RL 想让模型这么做，但 LLM 是否真的这么做了”。

### 2.2 闭环执行机制

`debate_rl_v2` 的主链路可以概括为：

```text
environment state
  -> observation encoding
  -> multi-agent RL controllers
  -> strategy bridge
  -> LLM agents
  -> transcript / decisions
  -> compliance verification
  -> reward computation
  -> policy update or online accumulation
  -> skill / causal / prompt distillation
```

对应到仓库中的核心实现：

- `debate_rl_v2/envs/fusion_env.py`
  融合环境主入口，负责一轮完整的 RL 引导辩论执行。
- `debate_rl_v2/core/strategy_bridge.py`
  将 RL 连续动作翻译为 LLM 可执行的策略参数。
- `debate_rl_v2/core/compliance_verifier.py`
  判断 LLM 响应是否符合 RL 控制意图。
- `debate_rl_v2/core/reward_design.py`
  将质量、分歧、合规等信号整合为奖励。
- `debate_rl_v2/mode/controller.py`
  管理训练模式和在线模式的行为差异。
- `debate_rl_v2/mode/online_updater.py`
  在在线模式下做无梯度参数累积。
- `debate_rl_v2/knowledge/distiller.py`
  从高质量 episode 中蒸馏经验。
- `debate_rl_v2/skills/skill_db.py`
  用 SQLite 持久化技能、因果链、提示词候选。

### 2.3 单轮推理过程

以 `debate_rl_v2/envs/fusion_env.py` 为例，一轮融合执行大致包含以下步骤：

1. 从当前环境状态编码共享观测与角色特定观测。
2. 每个 RL 控制器输出连续动作。
3. `StrategyBridge` 将动作翻译成温度、攻击性、细节程度、证据需求等策略信号。
4. LLM 智能体按角色生成提案、挑战、仲裁和协调文本。
5. `ComplianceVerifier` 检查生成文本与策略信号的一致性。
6. 奖励模块计算质量改进、分歧变化和合规奖励。
7. 根据模式决定是更新 RL，还是冻结 RL 并进行在线参数累积。
8. 将本轮高质量轨迹蒸馏为技能、因果链、提示词候选和长期记忆。

### 2.4 双模式机制

`debate_rl_v2/mode/controller.py` 明确区分两种模式：

- **Training 模式**
  RL 权重可更新，允许探索噪声，适合离线训练和策略预学习。
- **Online 模式**
  RL 主权重冻结，关闭探索，通过 `OnlineParameterUpdater` 做无梯度自适应，并持续沉淀经验资产。

这使系统具备“先训练，再部署；部署后不反向传播，但还能持续积累经验”的能力。

### 2.5 在线更新机制

在线模式的关键不是继续训练主网络，而是做轻量、稳健的参数积累。`debate_rl_v2/mode/online_updater.py` 中主要有两类机制：

- **指数滑动平均（EMA）**
  把本轮观测到的有效策略参数平滑地融入历史参数。
- **贝叶斯后验更新**
  在每个角色的参数维度上维护均值和方差，用质量反馈更新置信度。

最终，系统在“高置信维度”使用贝叶斯后验均值，在“尚未稳定维度”继续使用 EMA 参数。

### 2.6 经验沉淀机制

项目没有把高质量对话只当作日志保存，而是进一步沉淀为可检索资产：

- **skills**：可复用的辩论策略或协作策略。
- **causal_relations**：从轨迹中提取的因果关系。
- **prompt_candidates**：经进化保留下来的优质提示词模板。
- **memory insights**：跨轮次、跨任务可复用的长期记忆。

这些资产分别由：

- `debate_rl_v2/knowledge/distiller.py`
- `debate_rl_v2/skills/skill_db.py`
- `debate_rl_v2/prompt_evolution/evolver.py`
- `debate_rl_v2/memory/manager.py`

共同维护。

---

## 3. 系统架构

### 3.1 总体架构图

```text
                                   ┌────────────────────────────┐
                                   │        Config Layer         │
                                   │  YAML / dataclass config    │
                                   └──────────────┬─────────────┘
                                                  │
                         ┌────────────────────────┼────────────────────────┐
                         │                        │                        │
                         ▼                        ▼                        ▼
               ┌────────────────┐      ┌──────────────────┐      ┌──────────────────┐
               │ Numerical Env   │      │ LLM Debate Env    │      │ Fusion Debate Env │
               │ RL-only         │      │ LLM-only          │      │ RL + LLM         │
               └────────┬───────┘      └─────────┬────────┘      └─────────┬────────┘
                        │                        │                           │
                        │                        │                           ▼
                        │                        │                ┌─────────────────────┐
                        │                        │                │   Strategy Bridge    │
                        │                        │                │ action -> signals    │
                        │                        │                └──────────┬──────────┘
                        │                        │                           │
                        │                        └───────────────────────────┤
                        │                                                    ▼
                        │                                         ┌─────────────────────┐
                        │                                         │      LLM Agents      │
                        │                                         │ proposer/challenger  │
                        │                                         │ arbiter/coordinator  │
                        │                                         └──────────┬──────────┘
                        │                                                    │
                        ▼                                                    ▼
               ┌────────────────┐                                 ┌─────────────────────┐
               │ RL Algorithms   │<──────── rewards ───────────────│ Compliance + Reward │
               │ PPO / MADDPG    │                                 └──────────┬──────────┘
               └────────┬───────┘                                            │
                        │                                                     ▼
                        │                                         ┌─────────────────────┐
                        └──────── updates / frozen weights ───────│ Distill / Skill DB  │
                                                                  │ Prompt Evolution     │
                                                                  │ Online Updater       │
                                                                  └─────────────────────┘
```

### 3.2 模块职责

| 模块 | 目录 | 作用 |
|---|---|---|
| 环境层 | `debate_rl_v2/envs/` | 定义辩论、融合和文本环境，驱动完整 episode |
| 通用框架层 | `debate_rl_v2/framework/` | 将辩论逻辑抽象为更通用的多角色协作环境 |
| 智能体层 | `debate_rl_v2/agents/` | LLM Agent、MADDPG Agent、消息、Hook、Tracing |
| 算法层 | `debate_rl_v2/algorithms/` | MADDPG、经验回放、角色观测、训练工具 |
| 核心机制层 | `debate_rl_v2/core/` | 策略桥接、合规验证、奖励、对抗机制 |
| 模式控制层 | `debate_rl_v2/mode/` | Training / Online 双模式控制与在线参数累积 |
| 知识层 | `debate_rl_v2/knowledge/` | Episode 蒸馏与知识整合 |
| 技能层 | `debate_rl_v2/skills/` | SQLite 技能库、技能抽取、技能管理 |
| 因果层 | `debate_rl_v2/causal/` | 因果图、因果数据集、因果提取 |
| 提示词演化层 | `debate_rl_v2/prompt_evolution/` | 提示词种群维护、变异、交叉、适应度更新 |
| 场景层 | `debate_rl_v2/scenarios/` | 具体任务实现，如 `debate`、`code_review` |
| 基础设施层 | `debate_rl_v2/llm/`、`memory/`、`tools/`、`visualization/` | LLM 接入、记忆、工具调用、可视化 |

---

## 4. 项目树

### 4.1 仓库根目录

```text
debate_rl/
├── configs/                 # 根级 YAML 配置
├── checkpoints/             # 训练后的模型参数
├── figures/                 # 可视化结果图
├── scripts/                 # 训练、评估、测试、可视化脚本
├── tests/                   # 测试
├── debate_rl/               # 旧版主包（v1）
├── debate_rl_v2/            # 当前研究主线（v2）
├── mdt_game/                # MDT 游戏化场景相关代码
├── requirements.txt         # 依赖
├── pyproject.toml           # v2 打包配置
├── setup.py                 # v1 兼容打包入口
└── README.md                # 当前文档
```

### 4.2 `debate_rl_v2/` 重点目录

```text
debate_rl_v2/
├── agents/
│   ├── llm_agent.py
│   ├── maddpg_agent.py
│   ├── hooks.py
│   ├── tracing.py
│   └── context_compressor.py
├── algorithms/
│   ├── maddpg.py
│   ├── maddpg_trainer.py
│   ├── replay_buffer.py
│   └── role_observations.py
├── causal/
│   ├── dataset.py
│   ├── extractor.py
│   └── graph.py
├── config/
│   ├── master.py
│   ├── env.py
│   ├── llm.py
│   ├── rl.py
│   └── mechanisms.py
├── core/
│   ├── strategy_bridge.py
│   ├── compliance_verifier.py
│   ├── reward_design.py
│   ├── adversarial.py
│   ├── knowledge.py
│   └── soft_switch.py
├── envs/
│   ├── fusion_env.py
│   ├── llm_env.py
│   ├── debate_logic.py
│   └── event_emitter.py
├── framework/
│   ├── environment.py
│   ├── fusion.py
│   ├── scenario_builder.py
│   └── roles.py
├── knowledge/
│   ├── distiller.py
│   └── consolidator.py
├── mode/
│   ├── controller.py
│   └── online_updater.py
├── prompt_evolution/
│   ├── evolver.py
│   ├── evaluator.py
│   └── template_bank.py
├── scenarios/
│   ├── debate/
│   └── code_review/
├── skills/
│   ├── skill_db.py
│   ├── skill_extractor.py
│   └── skill_manager.py
├── llm/
├── memory/
├── tools/
├── visualization/
└── docs/
    ├── paper_appendix.md
    ├── neurips_draft_zh.md
    └── neurips_draft_en.md
```

### 4.3 推荐阅读顺序

如果你是为了理解项目而不是立刻运行，建议按这个顺序读代码：

1. `debate_rl_v2/README.md`
2. `debate_rl_v2/envs/fusion_env.py`
3. `debate_rl_v2/core/strategy_bridge.py`
4. `debate_rl_v2/core/compliance_verifier.py`
5. `debate_rl_v2/agents/llm_agent.py`
6. `debate_rl_v2/algorithms/maddpg_trainer.py`
7. `debate_rl_v2/mode/controller.py`
8. `debate_rl_v2/mode/online_updater.py`
9. `debate_rl_v2/knowledge/distiller.py`
10. `debate_rl_v2/framework/fusion.py`

---

## 5. 安装方式

### 5.1 环境要求

- Python `>= 3.10`
- PyTorch `>= 2.1`
- NumPy
- PyYAML
- OpenAI 兼容 SDK

可选依赖：

- `sentence-transformers`
  用于更强的语义检索或合规判断。
- `tiktoken`
  用于更准确的 token 估计。

### 5.2 安装

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

如果需要开发与可选组件：

```bash
pip install -e ".[dev,all]"
```

说明：

- `pyproject.toml` 目前以 `debate_rl_v2` 为主包。
- `setup.py` 仍保留旧版 `debate_rl` 的兼容入口。
- 如果你要研究最新架构，优先使用 `debate_rl_v2`。

---

## 6. 配置说明

常用配置文件位于根目录 `configs/`：

- `configs/default.yaml`
  旧版默认配置。
- `configs/default_v2.yaml`
  v2 默认配置。
- `configs/llm_debate.yaml`
  LLM 辩论配置示例。
- `configs/maddpg.yaml`
  MADDPG 训练配置。
- `configs/mdt_maddpg.yaml`
  MDT 融合配置。

`debate_rl_v2.config.load_config(path)` 会将 YAML 内容加载到 dataclass 配置对象中，并支持 `${ENV_VAR}` 环境变量替换。

---

## 7. 如何使用

### 7.1 最快的体验方式：运行 v2 的 Mock 案例

不需要 API key，直接验证整条 LLM 辩论链路：

```bash
python scripts/test_mdt_v2.py --mock
```

这个脚本会做以下事情：

1. 创建 v2 的 LLM Client。
2. 初始化四个角色的 `LLMAgent`。
3. 创建 Memory、ToolRegistry、HookManager、ContextCompressor。
4. 启动 `TextDebateEnv` 执行多轮 MDT 辩论。
5. 输出轮次摘要和最终结果。

适合你确认环境是否可用，以及快速理解 v2 的角色协作流程。

### 7.2 使用真实 LLM API 运行 v2

```bash
python scripts/test_mdt_v2.py \
  --provider custom \
  --base-url https://www.dmxapi.cn/v1 \
  --model qwen-plus \
  --api-key YOUR_KEY
```

或者使用配置文件：

```bash
python scripts/test_mdt_v2.py --config configs/llm_debate.yaml
```

### 7.3 以代码方式使用 v2 的 LLM-only 环境

```python
from debate_rl_v2 import load_config
from debate_rl_v2.llm import create_llm_client
from debate_rl_v2.agents.llm_agent import LLMAgent
from debate_rl_v2.envs.llm_env import TextDebateEnv

cfg = load_config("configs/default_v2.yaml")
client = create_llm_client(
    provider="mock",
    model="mock-1.0",
    temperature=0.7,
    max_tokens=2048,
)

agents = {
    role: LLMAgent(role=role, client=client)
    for role in ["proposer", "challenger", "arbiter", "coordinator"]
}

env = TextDebateEnv(
    topic="是否应当严格监管生成式 AI？",
    context="请从技术风险、产业影响和治理成本三个维度展开。",
    rules=[
        "必须给出清晰立场",
        "必须回应对方关键论点",
        "仲裁者需要给出结构化总结",
    ],
    max_rounds=4,
)

result = env.run(agents, verbose=True)
print(result["final_quality"])
```

### 7.4 以代码方式使用 v2 的 Fusion 环境

`FusionDebateEnv` 适合研究“RL 如何控制 LLM 协作行为”：

```python
from debate_rl_v2.core.strategy_bridge import StrategyBridge
from debate_rl_v2.envs.fusion_env import FusionDebateEnv
from debate_rl_v2.agents.llm_agent import LLMAgent
from debate_rl_v2.agents.maddpg_agent import MADDPGAgentGroup
from debate_rl_v2.llm import create_llm_client

client = create_llm_client(provider="mock", model="mock-1.0")

llm_agents = {
    role: LLMAgent(role=role, client=client)
    for role in ["proposer", "challenger", "arbiter", "coordinator"]
}

bridge = StrategyBridge()

rl_agents = MADDPGAgentGroup(
    obs_dims=bridge.obs_dims(),
    act_dims=bridge.act_dims(),
)

env = FusionDebateEnv(
    topic="自动驾驶责任划分",
    context="围绕技术责任、法规责任、企业责任展开。",
    bridge=bridge,
    max_rounds=5,
)

result = env.run(
    rl_agents=rl_agents,
    llm_agents=llm_agents,
    explore=False,
)

print(result["final_quality"])
```

### 7.5 训练旧版 RL 控制器

仓库根级训练脚本目前仍主要基于旧版 `debate_rl/`：

```bash
python -m scripts.train_maddpg --config configs/maddpg.yaml
```

这个脚本会：

1. 创建数值环境。
2. 根据 `StrategyBridge` 的观测维度和动作维度构建 MADDPG 控制器。
3. 使用经验回放和 `MADDPGTrainer` 进行训练。
4. 将 checkpoint 存到 `checkpoints/`。

### 7.6 运行旧版 Fusion 辩论

如果你已经有训练好的 MADDPG checkpoint，可以运行旧版融合脚本：

```bash
python -m scripts.run_fusion_debate \
  --config configs/maddpg.yaml \
  --checkpoint checkpoints/maddpg/final
```

注意：

- 这个脚本走的是旧版 `debate_rl/` 代码路径。
- 它对“如何把训练好的 RL 控制器接到真实 LLM 辩论上”仍然很有参考价值。
- 如果你研究最新抽象设计，建议结合 `debate_rl_v2/envs/fusion_env.py` 一起看。

---

## 8. 推荐使用路径

根据目标不同，推荐采用不同入口：

### 8.1 只是想快速验证项目可运行

直接执行：

```bash
python scripts/test_mdt_v2.py --mock
```

### 8.2 想理解 RL-LLM 融合原理

重点阅读：

- `debate_rl_v2/envs/fusion_env.py`
- `debate_rl_v2/core/strategy_bridge.py`
- `debate_rl_v2/core/compliance_verifier.py`
- `debate_rl_v2/mode/online_updater.py`

### 8.3 想跑训练与 checkpoint 流程

先使用旧版脚本：

```bash
python -m scripts.train_maddpg --config configs/maddpg.yaml
python -m scripts.run_fusion_debate --config configs/maddpg.yaml
```

### 8.4 想扩展到新任务

建议从 v2 通用框架路线开发：

1. 在 `debate_rl_v2/scenarios/` 中增加新场景。
2. 基于 `debate_rl_v2/framework/` 定义角色、观测、策略映射和奖励。
3. 复用 `LLMAgent`、`StrategyBridge`、`SkillDatabase` 和 `EpisodeDistiller`。

---

## 9. 输出物与中间结果

仓库中的典型输出包括：

- `checkpoints/`
  训练好的 MADDPG/PPO 等模型参数。
- `runs/`
  训练日志与实验输出。
- `figures/`
  仪表盘截图、机制演化图和报告图。
- `*.json`
  辩论结果、trace、dashboard 导出文件。
- `skills.db` 或自定义 SQLite 文件
  在线模式下沉淀的技能、因果链与提示词候选。

---

## 10. 研究价值与边界

### 10.1 适合做什么

这个仓库适合作为以下研究问题的实验平台：

- RL 引导的多智能体 LLM 协作
- 策略控制与自然语言生成解耦
- 生成行为合规性反馈
- 在线无梯度适应
- 技能持久化与长期经验积累
- 提示词进化与因果上下文增强

### 10.2 不适合如何理解

阅读和使用时需要注意以下边界：

- 当前仓库更像研究原型，不是完整产品。
- 根级脚本和包结构同时存在 v1 与 v2，两套入口并存。
- v2 的核心抽象更清晰，但并不是所有脚本都已经完全迁移到 v2。
- 在线模式中的“学习”主要是参数平滑和经验资产积累，不等价于在线反向传播更新主模型。

---

## 11. 相关文档

- `debate_rl_v2/README.md`
  v2 子项目说明。
- `debate_rl_v2/docs/paper_appendix.md`
  面向论文写作的项目原理附录。
- `debate_rl_v2/docs/neurips_draft_zh.md`
  中文论文草稿。
- `debate_rl_v2/docs/neurips_draft_en.md`
  英文论文草稿。

---

## 12. 一句话总结

如果只用一句话概括本仓库：

**它是一套把强化学习用于“控制多智能体语言协作策略”，再把大语言模型用于“执行自然语言推理与表达”的研究型融合框架。**

# debate_rl_v2

一个面向通用智能体的多角色辩论与协作强化学习框架。

`debate_rl_v2` 将多角色 LLM 协作、机制控制、知识约束和多智能体强化学习统一到同一个工程框架中，支持把“提议 - 质疑 - 仲裁 - 协调”的协作流程落地为可运行、可训练、可扩展的系统。

如果你希望先看研究导向、理论优先的说明，请阅读 [README.md](/root/myproject/adam/RL/debate_rl/debate_rl_v2/README.md)。本文件采用常规工程 README 的写法，重点回答：这个项目是什么、能做什么、怎么启动、代码在哪。

---

## 功能概览

- 四角色协作框架：`proposer`、`challenger`、`arbiter`、`coordinator`
- 动态对抗强度控制：根据分歧、分歧变化和时间压力调节 `lambda_adv`
- 概率软切换：在探索、标准讨论、仲裁介入之间平滑切换
- 魔鬼代言人验证：在接近共识时做鲁棒性检查，抑制虚假共识
- 分层强化学习：上层学协议控制，下层学角色策略
- 奖励与信用分配：支持过程奖励、终局奖励和 Shapley 校正
- 场景扩展：支持通用 debate、code review、MDT 等任务
- LLM 与 RL 融合：既可纯 LLM 协作，也可用 RL 产生控制信号

---

## 适用场景

- 需要多角色协作而不是单次生成的任务
- 存在规则、合规、证据或约束要求的任务
- 需要可解释讨论过程和最终结论的任务
- 希望把协作轨迹转化为长期知识资产的任务

不适合只需一次性文本生成、无需角色分工和过程控制的简单场景。

---

## 整体架构

```text
Task / Scenario
    ↓
GameScenario + GameEngine
    ↓
Observation / Strategy Bridge / Mechanisms
    ↓
Proposer / Challenger / Arbiter / Coordinator
    ↓
LLM output + Tool calls + State transition
    ↓
Reward / Credit Assignment / RL Update
    ↓
Knowledge / Memory / Skills / Causal assets
```

核心闭环如下：

1. 场景定义任务、状态、终止条件和奖励接口。
2. 四个角色根据观测和控制信号生成各自行为。
3. LLM 输出、工具结果和规则校验共同更新协作状态。
4. 机制层根据分歧和质量调节对抗节奏。
5. 奖励与信用分配回传给 RL 层，更新角色策略和元策略。

---

## 核心机制

### 1. 动态对抗强度

`core/adversarial.py`

根据当前分歧、分歧变化率和时间压力更新 `lambda_adv`，避免讨论一开始就过度对抗，也避免末期无限拖延。

### 2. 概率软切换

`core/soft_switch.py`

根据 `lambda_adv` 决定是否提高仲裁者介入概率，或重新增强挑战者的活跃度，避免阶段切换时抖动。

### 3. 魔鬼代言人验证

`core/devil_advocate.py`

当系统接近稳定时，发起额外强质疑测试。如果一轮强质疑就让系统重新发散，说明之前的共识并不稳健。

### 4. 奖励与信用分配

- `framework/reward.py`
- `scenarios/debate/reward.py`
- `algorithms/credit_assignment.py`

基础奖励关注质量、共识、合规和步长代价；场景奖励进一步加入 dense quality、curiosity 和边际收益惩罚；Shapley 机制用于更合理地给不同角色分配贡献。

### 5. 分层强化学习

`algorithms/hierarchical.py`

协调者负责上层元控制，其他角色负责下层任务行为。两层使用不同更新节奏，适合学习“说什么”和“何时切换机制”这两类不同问题。

---

## 项目结构

```text
debate_rl_v2/
├── algorithms/          # MAPPO、MADDPG、分层训练、信用分配
├── core/                # 对抗强度、软切换、魔鬼代言人、证据链等机制
├── framework/           # GameEngine、GameScenario、Reward、Observation 等抽象
├── scenarios/           # debate、code_review 等具体场景
├── knowledge/           # 规则、知识引擎、知识沉淀
├── memory/              # 长期记忆与轨迹资产
├── skills/              # 技能化协作模式
├── causal/              # 因果经验结构
├── prompt_evolution/    # 提示词候选演化
├── mode/                # training / online 等运行模式
├── agents/              # PPO/LLM 等角色代理
├── scripts/             # 测试、样例、运行脚本
└── visualization/       # 可视化与训练看板
```

---

## 安装

### 基础安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 开发依赖

```bash
pip install -e ".[dev,all]"
```

---

## 快速开始

### 1. 最快验证

```bash
cd /root/myproject/adam/RL/debate_rl
python3 scripts/test_mdt_v2.py --mock
```

这会用 mock 模式验证框架主流程是否可运行。

### 2. 使用真实 LLM

```bash
cd /root/myproject/adam/RL/debate_rl
python3 scripts/test_mdt_v2.py \
  --provider custom \
  --base-url https://api.example.com/v1 \
  --model qwen-plus \
  --api-key YOUR_KEY
```

### 3. 直接使用场景与引擎

```python
from debate_rl_v2.framework import ScenarioBuilder, GameToolRegistry
from debate_rl_v2.scenarios.debate import create_debate_scenario

scenario_config = create_debate_scenario()
builder = ScenarioBuilder(scenario_config)
tool_registry = GameToolRegistry()

engine = builder.create_game_engine(
    tool_registry=tool_registry,
    max_rounds=10,
)
```

---

## 常见使用方式

### 纯 LLM 协作

适合先验证角色分工、提示词、机制编排和任务闭环。

```python
from debate_rl_v2.framework import ScenarioBuilder
from debate_rl_v2.llm import create_llm_client
from debate_rl_v2.scenarios.debate import create_debate_scenario

client = create_llm_client(provider="mock", model="mock-1.0", temperature=0.7)
scenario_config = create_debate_scenario(
    topic="是否应当严格监管生成式 AI？",
    context="从技术风险、产业影响和治理成本三个维度展开。",
    max_rounds=4,
)
builder = ScenarioBuilder(scenario_config)
agents = builder.create_llm_agents(client)
engine = builder.create_game_engine(max_rounds=4)

result = engine.run_episode(agents, verbose=True)
print(result["final_quality"])
```

### RL 引导的 LLM 协作

适合在保留 LLM 表达能力的同时，用 RL 学习风格控制、强度控制和机制切换。

```python
from debate_rl_v2.framework import ScenarioBuilder
from debate_rl_v2.llm import create_llm_client
from debate_rl_v2.scenarios.debate import create_debate_scenario

client = create_llm_client(provider="mock", model="mock-1.0")
scenario_config = create_debate_scenario(
    topic="自动驾驶责任划分",
    context="围绕技术责任、法规责任、企业责任展开。",
    max_rounds=5,
)
builder = ScenarioBuilder(scenario_config)
agents = builder.create_llm_agents(client)
rl_agents = builder.create_maddpg_agents()
engine = builder.create_game_engine(max_rounds=5)

result = engine.run_episode(agents, rl_agents=rl_agents, explore=True)
print(result["reward_breakdown"])
```

---

## 关键模块说明

### 框架层

- `framework/game_scenario.py`
  定义场景接口、状态更新和终止条件。
- `framework/game_engine.py`
  负责驱动完整 episode。
- `framework/observation.py`
  负责将协作状态编码为 RL 可消费观测。
- `framework/reward.py`
  提供领域无关奖励基类。

### 机制层

- `core/adversarial.py`
  对抗强度控制器。
- `core/soft_switch.py`
  软切换控制器。
- `core/devil_advocate.py`
  鲁棒性验证控制器。
- `core/evidence_chain.py`
  证据链记录。
- `core/strategy_bridge.py`
  RL 信号到角色行为风格的桥接层。

### 学习层

- `algorithms/mappo.py`
  多智能体 PPO 更新。
- `algorithms/maddpg.py`
  多智能体 DDPG 变体。
- `algorithms/hierarchical.py`
  分层训练逻辑。
- `algorithms/credit_assignment.py`
  Shapley 信用分配。

### 场景层

- `scenarios/debate/`
  通用辩论场景。
- `scenarios/code_review/`
  代码审查场景。
- `mdt_game/`
  基于该框架构建的医学 MDT 示例工程。

---

## 内置场景

### Debate

标准四角色辩论场景：

- Proposer
- Challenger
- Arbiter
- Coordinator

```python
from debate_rl_v2.scenarios.debate import create_debate_scenario

scenario_config = create_debate_scenario()
```

### Code Review

代码审查场景：

- Author
- Reviewer
- Maintainer

```python
from debate_rl_v2.scenarios.code_review import create_code_review_scenario

scenario_config = create_code_review_scenario()
```

### MDT Game

`mdt_game` 是本框架的真实应用示例之一，用于展示如何把通用多角色协作、机制调控、奖励计算和报告生成落到高约束医学场景中。

参考：

- [README.md](/root/myproject/adam/RL/debate_rl/mdt_game/README.md)

---

## 开发建议

- 如果你在做新任务，优先从 `framework/` 和 `scenarios/` 扩展，而不是直接修改底层机制。
- 如果你在调讨论质量，优先检查 `reward.py`、`strategy_bridge.py`、`adversarial.py` 和场景奖励。
- 如果你在调收敛速度，优先看 `hierarchical.py`、`soft_switch.py` 和终止条件。
- 如果你在做真实产品适配，建议把任务规则和结构化输出先纳入 `scenario`，再接入真实 LLM。

---

## 已知说明

- 项目中部分旧训练入口仍为兼容保留，新的场景运行更推荐使用 `GameEngine + GameScenario`。
- 分层训练模块仍保留一定历史包袱，但其核心设计对理解整体框架仍然重要。
- 如果你关注理论与实现的一一映射，请优先阅读 [README.md](/root/myproject/adam/RL/debate_rl/debate_rl_v2/README.md)。

---

## 参考阅读

- 理论版说明：[README.md](/root/myproject/adam/RL/debate_rl/debate_rl_v2/README.md)
- MDT 示例：[README.md](/root/myproject/adam/RL/debate_rl/mdt_game/README.md)

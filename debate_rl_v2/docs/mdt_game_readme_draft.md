# mdt_game

`mdt_game` 是基于 `debate_rl_v2/framework` 构建的独立 MDT 多学科肿瘤讨论场景包。

它的定位不是一个通用聊天脚本，而是一个**可复用、可扩展、可评估的医学多角色协作场景实现**。你可以把它理解为 `debate_rl_v2` 通用博弈框架在医疗 MDT 场景中的一次完整落地。

如果 `debate_rl_v2` 回答的是“如何构建 RL-LLM 多智能体协作框架”，那么 `mdt_game` 回答的是：

**如何把这个框架具体化为一个带角色定义、患者生成、医学工具、状态更新和奖励计算的肿瘤 MDT 讨论任务。**

---

## 1. 项目定位

MDT 是肿瘤诊疗中的典型多角色协作场景。不同专科医生需要围绕同一患者病例，从不同专业视角提出方案、补充风险、评估质量并形成共识。

`mdt_game` 将这一过程建模为一个多角色协作 episode，核心要素包括：

- **患者病例**
  由合成患者生成器产生，包含分期、分子标志物、ECOG、合并症等信息。
- **专业角色**
  不同角色有不同阶段、输出结构和风格控制维度。
- **医学工具**
  支持指南查询、风险计算、临床试验搜索、药物交互检查。
- **状态更新**
  每轮根据各角色输出更新共识水平、质量得分和讨论元数据。
- **奖励评估**
  在 episode 结束后基于质量、指南符合度、安全性和共识情况计算奖励。

因此，`mdt_game` 不是单一 prompt 模板，而是一个完整的场景定义。

---

## 2. 系统原理

### 2.1 场景抽象

`mdt_game` 的核心依赖是 `debate_rl_v2.framework.game_scenario.GameScenario`。它把一个具体任务抽象成以下几个接口：

1. `setup()`
2. `register_tools()`
3. `create_episode()`
4. `build_role_prompt()`
5. `update_state()`
6. `check_terminal()`
7. `compute_rewards()`

`mdt_game/scenario.py` 中的 `MDTScenario` 实现了这一组接口，因此它可以直接交给 `GameEngine` 执行。

### 2.2 执行流程

在 `GameEngine` 路径下，一次 MDT episode 的流程是：

```text
create_episode
  -> 生成合成患者病例
  -> 构建 topic / patient_summary / 初始 metadata
  -> 按角色顺序构建 prompt
  -> LLM 角色执行
  -> update_state 更新讨论状态
  -> check_terminal 判断是否达成共识
  -> compute_rewards 使用工具做终局奖励计算
```

如果启用了 RL 控制器，则中间还会插入：

```text
observation encoder
  -> rl_agents.act()
  -> strategy_bridge.translate()
  -> style / temperature 注入各角色 agent
```

### 2.3 角色设计

当前内置四个 MDT 角色，定义在 `mdt_game/__init__.py` 中：

| 角色 | phase | 作用 |
|---|---|---|
| `oncologist` | `propose` | 肿瘤内科，负责制定系统治疗方案 |
| `surgeon` | `challenge` | 胸外科，评估手术可行性与风险 |
| `radiologist` | `challenge` | 放射科，评估影像学发现并提出放疗方案 |
| `pathologist` | `evaluate` | 病理科，综合评估方案质量并判断是否形成共识 |

这些角色不是简单字符串，而是 `RoleDefinition`，每个角色都显式包含：

- `description`
- `system_prompt`
- `output_schema`
- `style_dimensions`
- `action_dim`

这意味着 `mdt_game` 已经具备从“角色定义”直接驱动 LLM Agent 和 RL 控制器的能力。

### 2.4 患者建模

`mdt_game/patient_generator.py` 通过 `SyntheticPatientGenerator` 生成结构化病例，核心字段包括：

- 癌种与分期
- 年龄与性别
- 肿瘤大小与分级
- `ER` / `PR` / `HER2` / `BRCA`
- `ECOG` 与 Charlson 指数
- 合并症与既往治疗

患者对象 `PatientCase` 同时提供：

- `to_summary()`
  转成给角色阅读的自然语言病例摘要。
- `to_dict()`
  转成工具与观测编码器可消费的结构化字典。

另外，生成器还会给出一个 `target_treatment`，便于奖励函数做目标导向评估。

### 2.5 工具增强机制

`mdt_game/tools.py` 注册了四类医学工具，全部通过 `GameToolRegistry` 注入：

- `query_nccn_guidelines`
  查询模拟 NCCN 指南推荐。
- `calculate_risk_score`
  计算 Charlson / ECOG / TNM 风险评分。
- `search_clinical_trials`
  搜索模拟临床试验。
- `drug_interaction_check`
  检查药物相互作用。

这些工具既可以被 LLM 在多轮工具调用中使用，也可以被 `compute_rewards()` 在 episode 结束时用于奖励验证。

### 2.6 观测与策略桥接

如果在 `mdt_game` 中接入 RL，主要使用两个模块：

- `mdt_game/observation.py`
  `MDTObservationEncoder`
- `mdt_game/strategy.py`
  `MDTStrategyBridge` 和 `MDTStyleComposer`

当前观测设计为：

- `16D` 共享观测
- `4D` 角色扩展观测
- 总计 `20D` 角色输入

共享观测包含：

- 回合进度
- 质量分
- 共识水平
- 合规度
- 分期、年龄、ECOG、分子特征
- 方案数、质疑数、指南符合度、药物安全性

角色风格控制则根据专科语境做了专门设计，例如：

- `oncologist`
  `aggressiveness` / `evidence_reliance` / `guideline_adherence`
- `surgeon`
  `conservatism` / `detail_level` / `risk_tolerance`
- `radiologist`
  `thoroughness` / `specificity`
- `pathologist`
  `strictness` / `evidence_weight`

### 2.7 奖励计算

`MDTScenario.compute_rewards()` 不是简单返回质量分，而是综合了：

- 基础质量奖励
- 指南符合度 bonus
- 药物安全性 bonus
- 达成共识 bonus

其中指南符合度和安全性不是硬编码常量，而是通过工具调用进行验证，这使奖励逻辑更接近场景语义。

---

## 3. 架构说明

### 3.1 总体结构

```text
SyntheticPatientGenerator
  -> PatientCase
  -> MDTScenario.create_episode()
  -> GameEngine / DashboardGameRunner
  -> build_role_prompt()
  -> LLMAgent(role_definition=MDT_ROLES[role])
  -> ToolAugmentedAgentLoop / GameToolRegistry
  -> update_state()
  -> compute_rewards()
```

### 3.2 主要模块

| 模块 | 文件 | 作用 |
|---|---|---|
| 场景定义 | `scenario.py` | 定义 episode 创建、prompt 构建、状态更新、终止与奖励 |
| 角色注册 | `__init__.py` | 定义 MDT 角色、输出结构和一键工厂函数 |
| 患者生成 | `patient_generator.py` | 生成结构化肿瘤病例 |
| 观测编码 | `observation.py` | 将 MDT 状态编码为 RL 可用观测 |
| 策略桥接 | `strategy.py` | 将 RL 动作映射为 MDT 专业风格指令 |
| 医学工具 | `tools.py` | 指南、风险、试验、药物交互工具 |
| 仪表盘运行器 | `dashboard_runner.py` | 为实时面板复刻运行循环并推送事件 |
| 示例脚本 | `examples/`、`scripts/` | 演示不同运行路径 |

### 3.3 两条使用路径

`mdt_game` 当前支持两种理解和使用方式：

- **路径 A：通用博弈引擎主线**
  基于 `GameScenario + GameEngine`，这是更通用、更贴合 `framework/` 抽象的方式。
- **路径 B：TextDebateEnv 兼容示例线**
  复用 `debate_rl_v2` 已有辩论环境来跑具体 MDT 案例，适合演示和快速复现。

如果你是为了研究框架扩展能力，优先看路径 A。
如果你是为了快速体验多角色 MDT 文本协作，路径 B 更直接。

---

## 4. 项目树

```text
mdt_game/
├── __init__.py
├── scenario.py
├── observation.py
├── strategy.py
├── patient_generator.py
├── tools.py
├── dashboard_runner.py
├── examples/
│   ├── run_mdt_debate.py
│   ├── run_mdt_with_dashboard.py
│   └── visualize_mdt_result.py
├── scripts/
│   └── run_mdt_game.py
└── tests/
    └── test_mdt_game.py
```

### 4.1 推荐阅读顺序

建议按以下顺序阅读源码：

1. `__init__.py`
2. `scenario.py`
3. `patient_generator.py`
4. `tools.py`
5. `observation.py`
6. `strategy.py`
7. `scripts/run_mdt_game.py`
8. `dashboard_runner.py`
9. `tests/test_mdt_game.py`

---

## 5. 安装与依赖

`mdt_game` 不是一个完全脱离仓库的独立 PyPI 包，它依赖 `debate_rl_v2/framework`、`debate_rl_v2/agents` 和 `debate_rl_v2/llm`。

因此推荐在仓库根目录安装整个项目：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

常用依赖包括：

- `python >= 3.10`
- `torch`
- `numpy`
- `pyyaml`
- `openai`

如果需要图形面板：

- `PyQt5`
- `pyqtgraph`

---

## 6. 如何使用

### 6.1 最推荐的入口：通用博弈引擎

在仓库根目录执行：

```bash
python mdt_game/scripts/run_mdt_game.py --mock
```

这个脚本会：

1. 调用 `create_mdt_scenario()` 创建场景、角色注册表和工具注册表。
2. 为每个 MDT 角色创建一个 `LLMAgent(role_definition=...)`。
3. 使用 `GameEngine.run_episode()` 执行完整 episode。
4. 输出回合摘要、最终质量和各角色奖励。

如果你希望使用真实模型：

```bash
python mdt_game/scripts/run_mdt_game.py \
  --provider custom \
  --base-url https://www.dmxapi.cn/v1 \
  --api-key YOUR_KEY \
  --model qwen-plus
```

### 6.2 启用实时面板

```bash
python mdt_game/scripts/run_mdt_game.py --dashboard --mock
```

无 GUI 但生成静态报告：

```bash
python mdt_game/scripts/run_mdt_game.py --dashboard --no-gui --mock
```

这里内部使用的是 `DashboardGameRunner`，它会把 MDT 角色映射到 `LiveDebateDashboard` 的可视化语义：

- `oncologist -> proposer`
- `surgeon -> challenger`
- `radiologist -> challenger`
- `pathologist -> arbiter`

### 6.3 使用 TextDebateEnv 兼容案例

如果你想直接跑一个固定的肺癌 MDT 文本辩论案例，可以使用：

```bash
python mdt_game/examples/run_mdt_debate.py --mock
```

带实时面板：

```bash
python mdt_game/examples/run_mdt_with_dashboard.py --mock
```

这两份示例不是 `GameScenario` 主线，而是为了快速复现传统 MDT 辩论体验。

### 6.4 代码方式：最小创建场景

```python
from mdt_game import create_mdt_scenario

scenario, role_registry, tool_registry = create_mdt_scenario(seed=42)

episode = scenario.create_episode()
print(episode["topic"])
print(episode["patient_summary"])
```

### 6.5 代码方式：使用 GameEngine

```python
from debate_rl_v2.framework.game_engine import GameEngine
from debate_rl_v2.agents.llm_agent import LLMAgent
from debate_rl_v2.llm import create_llm_client
from mdt_game import MDT_ROLES, create_mdt_scenario

scenario, role_registry, tool_registry = create_mdt_scenario(seed=42)

client = create_llm_client(provider="mock", model="mock-1.0")

llm_agents = {
    role_name: LLMAgent(
        role=role_name,
        client=client,
        role_definition=role_def,
        game_tool_registry=tool_registry,
        max_tool_turns=3,
    )
    for role_name, role_def in MDT_ROLES.items()
}

engine = GameEngine(
    scenario=scenario,
    role_registry=role_registry,
    tool_registry=tool_registry,
    max_rounds=5,
)

result = engine.run_episode(llm_agents=llm_agents, verbose=True)
print(result["final_quality"])
```

### 6.6 代码方式：接入 RL 观测和策略桥

如果你要把 `mdt_game` 用作 RL-LLM 融合场景，可以配合：

- `MDTObservationEncoder`
- `MDTStrategyBridge`
- `MADDPGAgentGroup`

典型思路是：

1. `MDTObservationEncoder.encode_shared()` 生成共享状态。
2. `encode_role()` 生成角色输入。
3. RL 控制器输出连续动作。
4. `MDTStrategyBridge.translate()` 生成温度和风格维度。
5. `GameEngine` 将这些信号注入 LLM Agent。

这条路径更适合做实验和论文，而不是开箱即用脚本。

---

## 7. 输出结果

典型输出包括：

- `transcript`
  每轮各角色的结构化输出。
- `episode_context`
  当前患者病例和初始上下文。
- `rewards`
  每个 MDT 角色的终局奖励。
- `tool_context_log`
  奖励计算或推理阶段的工具调用记录。
- 静态报告图
  在 dashboard 模式下输出到 `figures/`。

---

## 8. 测试

`tests/test_mdt_game.py` 已覆盖以下部分：

- 场景工厂与角色注册
- 工具注册与 JSON 返回
- 合成患者生成器
- 观测编码器维度
- 场景端到端流程
- 工具增强 LLM 路径

在仓库根目录可以运行：

```bash
pytest mdt_game/tests/test_mdt_game.py
```

---

## 9. 适合的用途

`mdt_game` 适合用来做：

- 医疗多角色协作任务建模
- GameScenario 新场景开发示例
- RL-LLM 融合在医学讨论任务上的实验
- 工具增强多 Agent LLM 的结构化评估
- 论文中的具体场景案例章节

---

## 10. 边界说明

使用时需要注意：

- 当前医学工具是模拟实现，不是真实临床 API。
- 患者病例由合成生成器产生，不是临床真实数据。
- 奖励中的“指南符合度”和“药物安全性”是研究原型式近似，不应解释为真实医疗结论。
- 项目适合作为方法研究和框架验证，不适合直接用于临床决策支持。

---

## 11. 一句话总结

`mdt_game` 是 `debate_rl_v2` 通用多智能体协作框架在肿瘤 MDT 任务上的完整场景化实现，它把**患者、角色、工具、状态、奖励和可视化**都落成了可运行代码。

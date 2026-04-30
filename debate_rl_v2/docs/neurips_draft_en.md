# Dual-Mode Continual Adaptation for Multi-Agent Language Debate: A Closed-Loop Fusion of Reinforcement Learning and Large Language Models

## Authors

XXX¹, XXX¹, XXX¹, XXX², XXX¹†  
¹School of Computer Science, XXX University  
²School of Humanities and Social Sciences, XXX University  
†Corresponding author: xxx@xxx.edu.cn

## Abstract

Multi-agent large language models have shown strong promise in complex reasoning and structured collaboration, yet existing approaches still face three core limitations. First, role behavior is often driven by static prompts, providing little explicit and learnable high-level control over multi-agent interaction. Second, reinforcement learning is typically confined to offline training, making post-deployment adaptation expensive or impractical. Third, high-quality strategies, causal chains, and prompt patterns discovered during interaction are rarely consolidated into reusable long-term knowledge. We present a dual-mode continual adaptation method for multi-agent language debate and structured collaboration that unifies multi-agent reinforcement learning, large language model generation, execution-consistency feedback, and long-term experience accumulation within a single formulation. The key idea is to decouple high-level strategic control from natural language realization: a multi-agent controller outputs continuous actions, a strategy mapping function transforms these actions into temperature, stylistic, and mechanism-level control variables, and a large language model generates responses conditioned on these signals. We further introduce execution-consistency feedback that measures the discrepancy between intended control signals and realized language behavior, and feeds this discrepancy back into the optimization objective. On top of this closed loop, we define two complementary modes. In training mode, the model performs policy optimization, prompt evolution, causal extraction, and knowledge distillation. In online mode, the main policy is frozen, while no-gradient parameter adaptation, causal-context retrieval, and persistent skill memory enable continual improvement after deployment. Our contributions are threefold: (1) a unified dual-mode learning paradigm that combines offline policy optimization with online no-gradient adaptation; (2) a closed-loop path from policy control to language execution and consistency feedback; and (3) a continual evolution mechanism integrating causal reasoning, prompt evolution, and long-term persistent memory. The proposed formulation provides a principled basis for studying controllable, accumulative, and transferable multi-agent language collaboration.

## 1 Introduction

Large language models (LLMs) have made multi-agent language collaboration an increasingly important paradigm for solving complex tasks. By assigning different roles to different agents and organizing interaction over multiple rounds, multi-agent systems can exhibit complementary capabilities in analysis, critique, arbitration, and coordination. Debate is a particularly representative setting because it simultaneously requires stance expression, adversarial interaction, logical organization, evidence use, and dynamic strategic adjustment.

Despite recent progress, most existing multi-agent LLM methods still rely on static prompts and hand-crafted protocols. Such systems can induce some degree of role specialization, but they provide limited means for explicitly learning and optimizing the collaboration process itself. Reinforcement learning offers a principled framework for optimizing sequential decision making, yet most RL-for-LLM research focuses on single-model alignment or offline optimization and pays limited attention to role-differentiated control, post-deployment adaptation, and long-term experience accumulation in multi-agent language interaction.

This paper studies the following problem: how to build a multi-agent language debate system that supports explicit strategic control, verifiable language execution, low-cost continual adaptation after deployment, and reusable long-term experience across tasks. To address this problem, we propose a dual-mode continual adaptation method that integrates multi-agent reinforcement learning, language generation, execution-consistency feedback, and persistent experience accumulation.

The central idea is to explicitly decouple high-level strategic control from low-level natural language realization. Instead of generating text directly, the policy learns low-dimensional and interpretable continuous control actions. A strategy mapping module then transforms these actions into temperature, stylistic dimensions, and mechanism-level parameters for the language model. To ensure that intended strategies are actually reflected in generated language, we introduce execution-consistency feedback, which measures the alignment between intended control signals and realized responses and incorporates this alignment into the optimization objective. Furthermore, by combining a training mode and an online mode within a unified execution process, the method supports both offline policy learning and post-deployment no-gradient continual adaptation.

Our contributions are summarized as follows:

1. We propose a unified dual-mode learning framework that combines offline reinforcement learning with online no-gradient continual adaptation.
2. We introduce a strategy-to-language mapping mechanism that enables explicit high-level control over multi-agent language collaboration.
3. We incorporate execution-consistency feedback into the optimization loop, thereby closing the gap between strategic intent and language realization.
4. We integrate causal reasoning, prompt evolution, and persistent skill memory into a continual evolution mechanism for long-term experience accumulation.

## 2 Related Work

### 2.1 Multi-Agent Large Language Models

Multi-agent LLM systems have recently been used in collaborative reasoning, planning, medical discussion, and software engineering. By splitting roles and coordinating interactions across multiple turns, such systems often outperform single-agent methods on complex tasks. However, most of them still depend on fixed prompts and manually designed workflows, offering limited explicit control over interaction dynamics.

### 2.2 Reinforcement Learning for Language Models

Reinforcement learning has been widely studied for language model alignment, preference optimization, and generation control. Yet much of the existing literature focuses on single-model optimization or offline training. In contrast, this work addresses role-differentiated strategic control and continual adaptation in multi-agent language interaction.

### 2.3 Automatic Debate and Structured Adversarial Reasoning

Automatic debate systems aim to let models argue for or against a given proposition. Early work relied on rules, retrieval, or templates, while recent LLM-based approaches have significantly improved fluency and diversity. Nevertheless, explicit strategic control, execution verification, and persistent experience accumulation remain underexplored.

### 2.4 Prompt Optimization, Causal Reasoning, and Long-Term Memory

Prompt optimization, causal reasoning, and memory mechanisms have each been studied as ways to improve language models on complex tasks. However, relatively few works place them under a unified learning process for multi-agent collaboration. Our work combines these components within a continual adaptation framework.

## 3 Problem Formulation

### 3.1 Multi-Role Language Collaboration

Given an input task \(x\), a role set \(R=\{r_1,\dots,r_n\}\), interaction history \(H_t\), and system state \(s_t\), the goal is to generate structured responses \(y_i^t\) for each role at round \(t\), and eventually produce a final collaborative outcome \(Y\). The system seeks to optimize task quality, collaboration efficiency, and execution consistency.

For role \(i\), we define its local observation, action, and reward as

$$
o_i^t = Enc_i(s_t,H_t), \qquad
a_i^t = \pi_i(o_i^t), \qquad
r_i^t = R_i(s_t,s_{t+1},y^t).
$$

In the debate setting, the local observation can be decomposed into shared and role-specific components:

$$
o_i^t = \left[o_{\mathrm{shared}}^t; o_{\mathrm{role},i}^t\right].
$$

### 3.2 Decoupling Strategic Control from Language Realization

Rather than generating language directly, the controller learns continuous control actions that are mapped into language-facing control signals:

$$
z_i^t = B_i(a_i^t),
$$

where \(z_i^t\) may consist of temperature, stylistic, and mechanism-level variables:

$$
z_i^t = \left(\tau_i^t, s_i^t, m_i^t\right).
$$

The language model then generates a response conditioned on these control variables:

$$
y_i^t = \mathcal{L}_i(x,H_t,z_i^t).
$$

The overall objective is

$$
\max_{\Pi}\; \mathbb{E}\left[\sum_{t=1}^{T}\sum_{i=1}^{n} r_i^t \right].
$$

## 4 Method

### 4.1 Dual-Mode Learning

We define two complementary learning modes on top of a shared execution process:

- Training mode updates policy parameters while enabling prompt evolution, causal extraction, and knowledge distillation.
- Online mode freezes the main policy and performs no-gradient adaptation using persistent memory and new interaction feedback.

Formally, let \(M\) denote the current mode. The policy update rule is

$$
Update(M)=
\begin{cases}
1, & M=\text{training}, \\
0, & M=\text{online}.
\end{cases}
$$

The exploration noise is

$$
Noise(M)=
\begin{cases}
\epsilon, & M=\text{training}, \\
0, & M=\text{online}.
\end{cases}
$$

This formulation allows the system to maintain a stable learned controller while still adapting after deployment.

### 4.2 Strategy-to-Language Mapping

The controller outputs continuous actions in a low-dimensional strategic space, and a strategy mapping module transforms these actions into interpretable control variables for the language model. In debate, these control variables may correspond to confidence, detail level, aggressiveness, constructiveness, strictness, or coordination parameters. More generally, the mapping can be viewed as

$$
B_i:\mathcal{A}_i \rightarrow \mathcal{Z}_i.
$$

This separation between action space and language space makes the method more interpretable and easier to optimize.

### 4.3 Execution-Consistency Feedback

Without explicit feedback on whether generated language follows intended strategic signals, the system becomes effectively open-loop. We therefore introduce execution-consistency feedback. Let \(c_i^t \in [0,1]\) denote the consistency score between the intended control signal \(z_i^t\) and the realized response \(y_i^t\). The reward becomes

$$
r_i^t = r_{0,i}^t + \lambda c_i^t,
$$

where \(r_{0,i}^t\) is the base reward and \(\lambda\) is a weighting coefficient.

More generally, we use a layered reward decomposition:

$$
r_i^t = r_{\mathrm{task}}^t + r_{\mathrm{process},i}^t + r_{\mathrm{role},i}^t + r_{\mathrm{comp},i}^t + r_{\mathrm{terminal},i}^t.
$$

This decomposition allows the optimization process to distinguish global task progress, role-specific behavior, and execution fidelity.

### 4.4 Causal Context and Long-Term Skill Memory

To improve structured reasoning, we incorporate causal context and persistent skill memory. Let \(c(x)\) denote retrieved causal context and \(k(x)\) denote retrieved skill memory for input \(x\). The effective language-model input becomes

$$
\tilde{x}_i^t = \left[x, H_t, z_i^t, c(x), k(x)\right].
$$

This makes current generation depend not only on immediate control signals but also on accumulated long-term experience.

### 4.5 Prompt Evolution

To avoid fixed prompts, we maintain a population of prompts for each role. Let \(\mathcal{P}_i^{(g)}\) denote the prompt population of role \(i\) at generation \(g\), and let \(\mathcal{F}_i^{(g)}\) denote its fitness values. The evolution step is

$$
\mathcal{P}_i^{(g+1)} = \mathcal{E}\left(\mathcal{P}_i^{(g)}, \mathcal{F}_i^{(g)}\right),
$$

where \(\mathcal{E}\) is the evolutionary operator composed of selection, mutation, and crossover. This turns prompts into adaptive strategy carriers rather than fixed templates.

### 4.6 Online No-Gradient Adaptation

In online mode, the main controller is frozen, while role parameters are adapted without backpropagation. Let \(\theta_{t-1}\) be the previous parameter vector and \(\hat{\theta}_t\) the current observed parameter vector. Exponential smoothing gives

$$
\theta_t = (1-\alpha)\theta_{t-1} + \alpha \hat{\theta}_t,
$$

where \(\alpha \in (0,1)\) is the smoothing coefficient.

Under a posterior-estimation view, parameter refinement follows

$$
p(\theta \mid \mathcal{D}_t) \propto p(\mathcal{D}_t \mid \theta)\, p(\theta),
$$

where \(\mathcal{D}_t\) denotes new interaction feedback. The final control parameter is written as

$$
z_i^{t,*} = \hat{z}_i^t + \Delta z_i^t,
$$

where \(\hat{z}_i^t\) is the frozen controller output and \(\Delta z_i^t\) is the online correction. This enables low-cost continual adaptation without online policy-gradient updates.

## 5 Experimental Setup

### 5.1 Experimental Questions

The proposed method is best presented as a research method and experimental framework, so this section specifies an evaluation protocol consistent with the formulation rather than reporting fixed empirical numbers in the absence of completed experiments. The central questions are:

1. Does dual-mode learning outperform single-stage training?
2. Does execution-consistency feedback improve controllability?
3. Do causal context, persistent skill memory, and prompt evolution provide complementary gains?
4. Can online no-gradient adaptation improve performance while keeping the main controller frozen?

### 5.2 Evaluation Metrics

Recommended metrics include:

1. final quality score,
2. consensus rate,
3. average number of rounds,
4. execution-consistency score,
5. logical consistency or causal-chain hit rate,
6. long-horizon trend under online adaptation.

### 5.3 Baselines and Ablations

Recommended baselines include:

- a single-agent LLM,
- a multi-agent fixed-prompt system,
- an RL+LLM variant without consistency feedback,
- a dual-mode variant without online adaptation.

At minimum, the following ablations should be considered:

1. removing dual-mode learning,
2. removing execution-consistency feedback,
3. removing role-specific observations,
4. removing causal-context injection,
5. removing persistent skill memory,
6. removing prompt evolution,
7. removing online no-gradient adaptation.

## 6 Limitations

The proposed method still has several limitations. First, debate remains the primary validation setting, and broader evaluation across structured collaboration tasks is still needed. Second, the current consistency-feedback implementation still partially relies on heuristic signals; stronger semantic evaluators could improve robustness. Third, online adaptation in this work is based on no-gradient parameter refinement and persistent memory rather than full online policy optimization, so its long-term theoretical behavior remains to be studied. Finally, the quality of causal chains and persistent skills is still bounded by the capabilities of the underlying language model.

## 7 Broader Impact

This work aims to improve controllability, accumulation, and transferability in multi-agent language collaboration, which may be beneficial for debate systems, educational support, collaborative analysis, and structured decision assistance. At the same time, adversarial language systems may also be used to generate more persuasive manipulation, misleading arguments, or amplified biases. Future research and deployment should therefore enforce clear boundaries for high-risk domains, include content-safety and human-oversight mechanisms, and evaluate not only task performance but also safety, fairness, and misuse risks.

## 8 Conclusion

We presented a dual-mode continual adaptation method for multi-agent language debate and structured collaboration. The method unifies multi-agent reinforcement learning, language generation, execution-consistency feedback, and long-term experience accumulation in a single formulation. By explicitly controlling high-level strategy, closing the loop between intended and realized behavior, and supporting post-deployment adaptation through no-gradient updates and persistent memory, the method offers a principled basis for studying controllable, accumulative, and transferable multi-agent language collaboration.

## References

References should be completed in NeurIPS style and should prioritize work on:

- multi-agent LLM systems,
- reinforcement learning for language model optimization,
- automatic debate,
- prompt optimization and evolution,
- causal reasoning and long-term memory,
- multi-agent reinforcement learning.

"""
Observe 节点 — Agent 的"眼睛"和"耳朵"
=======================================

这个节点是 LangGraph 执行流程的第一步，负责"观察"当前状态，
为 LLM 推理准备好完整的上下文。

====================================
Observe 节点的核心职责
====================================

如果说 Think 是大脑，Act 是双手，那么 Observe 就是眼睛和耳朵。
它的工作是：

1. 检查是否需要压缩对话历史（防止超出 token 限制）
2. 准备 System Prompt（系统指令），告诉 LLM 它是谁、能做什么
3. 注入记忆和技能信息，让 LLM 了解上下文
4. 将 SystemMessage 放到消息列表的最前面

====================================
System Prompt 详解
====================================

System Prompt 是发给 LLM 的"角色设定"和"游戏规则"。它位于消息列表
的最前面（SystemMessage），告诉 LLM：

  - 它的角色是什么（"你是一个通用的 AI 助手"）
  - 它有哪些工具/技能可用
  - 用户的相关记忆是什么
  - 它应该遵循什么行为准则

在本实现中，System Prompt 由以下几部分拼接而成：

  SYSTEM_PROMPT_BASE（基础模板）
  + compact_summary（历史摘要，如果有的话）
  + skill instructions（活跃技能指令，如果有的话）

====================================
为什么只在没有 SystemMessage 时才注入？
====================================

这是 Observe 节点的一个关键设计决策：

  has_system = any(isinstance(m, SystemMessage) for m in messages)
  if not has_system:
      # 注入 SystemMessage

原因：
  在单次图执行中，Observe → Think → Act → Observe 会形成循环。
  每次回到 Observe 时，如果重新注入 SystemMessage，会导致：
  - 消息列表中出现多个 SystemMessage（重复，浪费 token）
  - System Prompt 和消息的相对顺序被打乱

  所以只在第一轮（还没有 SystemMessage 时）注入。
  后面的循环中，SystemMessage 一直保持在列表最前面。

  类比：就像话剧开场时介绍角色和背景，只需要在开场说一次，
  不需要每句台词前都重复一遍"我是谁，我在哪，我要干什么"。

====================================
config 参数的作用
====================================

config 是 LangGraph 的标准参数，通过 config["configurable"] 传递
自定义配置。本项目中用它传递：

  - tool_registry: 工具注册表
  - skill_registry: 技能注册表
  - model: LLM 模型实例

这是 LangGraph 推荐的依赖注入方式 — 不把外部依赖硬编码在节点函数中，
而是通过 config 传递。这样同一个图可以用不同的配置运行（比如测试时
用 mock 模型，生产时用真实模型）。

config 通过 graph.ainvoke(input, config) 传入：
  graph.ainvoke(
      {"messages": [HumanMessage(content="你好")]},
      {"configurable": {"model": my_model, "tool_registry": registry}}
  )
"""

from langchain_core.messages import SystemMessage
from src.agent.state import AgentState
from src.agent.compact import should_compact, compact_messages, estimate_messages_tokens
from config.settings import COMPACT_THRESHOLD_RATIO, COMPACT_KEEP_RECENT


# ---------------------------------------------------------------------------
# System Prompt 基础模板
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """你是一个通用的 AI 助手。你可以使用工具来完成用户的任务。

## 可用技能
{skill_summaries}

## 记忆
{recalled_memories}

## 行为准则
- 仔细分析用户的需求后再行动
- 优先使用技能（如果匹配），技能会提供详细的执行指导
- 如果无法完成，请诚实告知用户原因
"""


def format_skill_summaries(summaries: list[dict]) -> str:
    """
    格式化技能摘要列表为 Markdown 格式的字符串。

    每个技能是一个包含 name 和 summary 的字典：
      {"name": "mysql", "summary": "MySQL 数据库查询技能"}

    格式化后的示例：
      - **mysql**: MySQL 数据库查询技能
      - **redis**: Redis 缓存操作技能

    如果没有技能，返回占位文本。

    Parameters
    ----------
    summaries : list[dict]
        技能摘要列表，每个元素是 {"name": str, "summary": str}

    Returns
    -------
    str
        Markdown 格式的技能列表字符串
    """
    if not summaries:
        return "（暂无可用技能）"
    lines = []
    for s in summaries:
        lines.append(f"- **{s['name']}**: {s['summary']}")
    return "\n".join(lines)


def format_recalled_memories(memories: list[dict]) -> str:
    """
    格式化召回的记忆列表为 Markdown 格式的字符串。

    记忆系统会根据用户当前输入，检索出相关的历史记忆。
    每条记忆包含 content（内容）和 category（分类）。

    格式化后的示例：
      - [preference] 用户喜欢简洁的回答
      - [fact] 用户的姓名是张三

    这些信息帮助 LLM 提供个性化的回复。

    Parameters
    ----------
    memories : list[dict]
        记忆列表，每个元素是 {"content": str, "category": str}

    Returns
    -------
    str
        Markdown 格式的记忆列表字符串
    """
    if not memories:
        return "（暂无相关记忆）"
    lines = []
    for m in memories:
        cat = m.get("category", "fact")
        lines.append(f"- [{cat}] {m['content']}")
    return "\n".join(lines)


async def observe_node(state: AgentState, config: dict = None) -> dict:
    """
    Observe 节点：准备 LLM 推理所需的完整上下文。

    这个节点是图的入口，每次进入时执行以下操作：

    1. 【压缩检查】如果消息太长（token 数超过阈值），将旧消息压缩为摘要
    2. 【注入 System Prompt】如果消息列表还没有 SystemMessage，插入系统指令
    3. 【注入技能信息】在 System Prompt 中列出可用技能
    4. 【注入记忆】在 System Prompt 中列出与当前对话相关的记忆
    5. 【注入活跃技能指令】如果有技能被激活，在 System Prompt 中追加指令

    === 处理流程 ===

    ┌─────────────────────────────────────────────────────┐
    │  1. 读取当前 messages 列表和 compact_summary       │
    │  2. 检查是否需要压缩（should_compact）              │
    │     ├─ 需要且未压缩 → 压缩旧消息 → 更新 messages  │
    │     └─ 不需要       → 跳过                         │
    │  3. 检查 messages 中是否有 SystemMessage           │
    │     ├─ 已有 → 不做任何事（避免重复注入）           │
    │     └─ 没有 → 构造 SystemMessage，放到列表最前面   │
    │  4. 返回更新后的状态                               │
    └─────────────────────────────────────────────────────┘

    Parameters
    ----------
    state : AgentState
        当前 Agent 状态，包含 messages, compact_summary, recalled_memories,
        active_skill 等字段。

    config : dict, optional
        LangGraph 的 config 对象，通过 config["configurable"] 获取：
        - skill_registry: 技能注册表
        - model: LLM 模型实例（用于压缩时生成摘要）

    Returns
    -------
    dict
        部分状态更新字典。可能包含：
        - messages: 更新后的消息列表（追加了 SystemMessage 或压缩后替换）
        - compact_summary: 压缩摘要
    """
    # 获取当前消息列表
    messages = state.get("messages", [])
    compact_summary = state.get("compact_summary")
    recalled = state.get("recalled_memories", [])
    result = {}

    # ------------------------------------------------------------------
    # Step 1: 上下文压缩检查
    #
    # 当消息历史很长时（比如多轮工具调用后），token 数可能超出 LLM 的
    # 上下文窗口限制。需要将早期消息压缩为一段摘要，保留最近的消息。
    #
    # should_compact 使用 tiktoken 估算当前消息的 token 数，
    # 如果超过 max_tokens * threshold_ratio（默认 65536 * 0.8 = ~52428），
    # 则触发压缩。
    #
    # 注意：只在 compact_summary 为 None 时才执行压缩（避免重复压缩）。
    # 如果已经有摘要了，说明之前压缩过，直接复用即可。
    # ------------------------------------------------------------------
    if should_compact(messages, COMPACT_THRESHOLD_RATIO, max_tokens=65536) and compact_summary is None:
        # 从 config 中获取 LLM 模型实例（用于生成摘要）
        model = config.get("configurable", {}).get("model") if config else None
        if model:
            # compact_messages 返回：
            #   summary: 压缩后的摘要文本
            #   recent:  最近 N 条消息（保留原文，不压缩）
            summary, recent = await compact_messages(messages, model, COMPACT_KEEP_RECENT)
            result["compact_summary"] = summary  # 保存摘要到状态
            messages = recent                     # 用最近消息替换原列表
            compact_summary = summary             # 更新局部变量，后续使用

    # ------------------------------------------------------------------
    # Step 2: 获取技能摘要
    #
    # 从 skill_registry 获取所有注册技能的简介，格式化为列表。
    # 这些信息会被注入到 System Prompt 中，让 LLM 知道自己有哪些
    # "特殊能力"（技能）可以使用。
    # ------------------------------------------------------------------
    skill_summaries = []
    if config and config.get("configurable", {}).get("skill_registry"):
        skill_summaries = config["configurable"]["skill_registry"].get_summaries()

    # ------------------------------------------------------------------
    # Step 3: 检查 SystemMessage 是否已存在
    #
    # 这是关键判断：如果消息列表里已经有 SystemMessage 了，
    # 说明这不是第一轮（可能刚从 Act 循环回来），不需要重复注入。
    #
    # 为什么用 isinstance 检查？
    #   messages 列表中可能有多种类型的消息，如 HumanMessage,
    #   AIMessage, ToolMessage。SystemMessage 是 LangChain 中
    #   专门用来表示"系统指令"的消息类型。
    # ------------------------------------------------------------------
    has_system = any(isinstance(m, SystemMessage) for m in messages)

    if not has_system:
        # --------------------------------------------------------------
        # Step 4: 构造 System Prompt
        #
        # 使用 Python 字符串的 .format() 方法，用实际数据填充模板中的
        # {skill_summaries} 和 {recalled_memories} 占位符。
        # --------------------------------------------------------------
        system_text = SYSTEM_PROMPT_BASE.format(
            skill_summaries=format_skill_summaries(skill_summaries),
            recalled_memories=format_recalled_memories(recalled),
        )

        # --------------------------------------------------------------
        # Step 5: 前置历史摘要（如果有的话）
        #
        # 如果存在压缩摘要（即之前的对话被压缩了），
        # 把摘要放在 System Prompt 的最前面。
        # 这样 LLM 首先看到"之前发生了什么"，然后是"我是谁/能做什么"。
        # --------------------------------------------------------------
        if compact_summary:
            system_text = f"## 历史对话摘要\n{compact_summary}\n\n{system_text}"

        # --------------------------------------------------------------
        # Step 6: 注入活跃技能指令（Fix 4）
        #
        # 当用户触发了某个技能时（active_skill 非空），
        # 在 System Prompt 末尾追加该技能的详细指令和工具白名单。
        #
        # 示例输出：
        #   ## 当前激活技能: mysql
        #   你可以使用 MySQL 查询数据库。请先理解用户需求再编写 SQL。
        #   允许的工具: execute_sql, list_tables, describe_table
        # --------------------------------------------------------------
        if state.get("active_skill"):
            skill = state["active_skill"]
            skill_instructions = (
                f"\n## 当前激活技能: {skill['name']}\n"
                f"{skill['description']}\n"
                f"允许的工具: {', '.join(skill['tools'])}"
            )
            system_text += skill_instructions

        # --------------------------------------------------------------
        # Step 7: 将 SystemMessage 插入消息列表最前面
        #
        # 这是最终的消息列表结构（从上到下）：
        #   [0] SystemMessage    ← 系统指令（刚创建的）
        #   [1] HumanMessage     ← 用户输入
        #   [2] AIMessage        ← AI 回复（之前轮次）
        #   [3] ToolMessage      ← 工具结果（之前轮次）
        #   ...
        #
        # 注意：list(messages) 创建了一个新列表，
        # 确保不会修改原有的 messages 引用。
        # --------------------------------------------------------------
        new_messages = [SystemMessage(content=system_text)] + list(messages)
        result["messages"] = new_messages
        return result

    # 已有 SystemMessage → 不做任何修改，返回空字典
    return result

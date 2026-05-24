"""
Agent 图构建模块（graph.py）
============================

这个模块是 Agent 的"大脑结构图" — 它定义了 Agent 执行流程的拓扑结构。

====================================
LangGraph 核心概念：StateGraph
====================================

StateGraph 是 LangGraph 框架的核心类。它让你把 Agent 的执行流程定义为一个
**有向图** (Directed Graph)，其中：

  - 节点 (Node) = 处理步骤（函数），读入 State，返回部分 State
  - 边 (Edge)   = 节点间的流转方向
  - 状态 (State) = 在图中流转的共享数据（就是我们的 AgentState）

为什么要用图来建模 Agent？

  传统的链式调用（A → B → C）太僵硬了。真正的 Agent 需要：
  - 根据条件走不同分支（"需要工具就执行，不需要就结束"）
  - 循环（"观察→思考→执行→观察..."直到任务完成）
  - 在任意节点暂停和恢复（checkpoint 持久化）

  有向图完美支持这些需求，而且 LangGraph 让它像搭积木一样简单。

====================================
Observe → Think → Act 循环
====================================

这是本 Agent 的核心执行循环，灵感来自认知科学中的 OODA 循环
（Observe, Orient, Decide, Act），简化为三步：

            ┌──────────────────────────────────┐
            │                                  │
            ▼                                  │
   ┌──────────┐    ┌──────────┐    ┌──────────┐│
   │ Observe  │───▶│  Think   │───▶│   Act    ││
   │ 观察     │    │  思考    │    │  行动    ││
   └──────────┘    └──────────┘    └──────────┘│
                          │                     │
                          │ final_answer 非空   │
                          ▼                     │
                        ┌────┐                  │
                        │END │（结束，不循环）   │
                        └────┘                  │
                          │                     │
                          │ tool_calls 非空     │
                          └─────────────────────┘
                                （回到 Act，循环）

  1. Observe（观察）：准备上下文 — 注入 System Prompt、召回记忆、
                      处理消息压缩
  2. Think（思考）：调用 LLM 推理 — 决定是直接回答还是调用工具
  3. Act（行动）：执行 LLM 请求的工具调用 — 将结果注入消息列表

  关键设计：Observe 和 Act 之间形成循环
    Act 执行完工具后 → 回到 Observe → Observe 注入新的上下文 →
    Think 重新评估 → 可能再次调用工具或给出最终答案

    这就是 Agent 的"自主循环"能力：它可以多次调用工具，
    每次根据工具返回的结果调整下一步策略，直到能给出最终答案。

====================================
Checkpoint（检查点）是什么？
====================================

Checkpoint 是 LangGraph 的持久化机制，相当于游戏中的"存档点"。

  为什么需要它？
  - 多轮对话：用户说完一句话后，Agent 需要记住前面的对话历史。
    如果没有 checkpoint，每次用户发消息，Agent 都像是"失忆"了。

  - 断点续传：如果 Agent 在执行过程中崩溃或超时，checkpoint 允许
    从上次中断的地方恢复，而不是从头开始。

  - 人机协作：在某些设计中，可以在关键步骤设置"人工审核点"，
    Agent 暂停等待人类确认后再继续。

  工作原理：
    每经过一个节点（或每执行一个步骤），LangGraph 自动将当前的
    AgentState 序列化保存到 checkpointer 中（可以是 SQLite 数据库、
    内存、或自定义存储）。下次同一个 session_id 的请求进来时，
    LangGraph 自动恢复上次的状态。

  类比：就像你玩 RPG 游戏，每次进入新的地图自动存档。
  下次打开游戏，你从上次离开的地方继续，而不是从第一关重新开始。

  常见用法：
    - MemorySaver: 存在内存中，适合开发测试（重启丢失）
    - SqliteSaver: 存在 SQLite 文件中，适合生产环境（持久化）
"""

from typing import Literal
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes.observe import observe_node
from src.agent.nodes.think import think_node
from src.agent.nodes.act import act_node


# ---------------------------------------------------------------------------
# 条件路由函数 — Think 之后去哪里？
# ---------------------------------------------------------------------------

def route_after_think(state: AgentState) -> Literal["act", "__end__"]:
    """
    Think 节点之后的条件路由：决定下一步是"执行工具"还是"结束对话"。

    这个函数是 add_conditional_edges 的回调函数。
    LangGraph 在 Think 节点执行完后调用它，根据返回值决定下一步：

    === 路由逻辑 ===

    1. 如果 final_answer 不为 None：
       → 返回 "__end__"（LangGraph 内部常量 END 的字符串表示）
       → 这意味着 LLM 已经给出了最终回答，不需要再调用工具
       → 图执行结束，final_answer 的内容返回给用户

    2. 如果 tool_calls 列表非空：
       → 返回 "act"
       → 这意味着 LLM 决定需要调用工具来获取更多信息
       → 流程进入 Act 节点，执行工具调用

    3. 其他情况（tool_calls 为空且 final_answer 也为空）：
       → 返回 "__end__"
       → 算是兜底处理：LLM 既没说要回答也没说要调用工具
       → 安全起见，直接终止

    === 返回值类型说明 ===

    Literal["act", "__end__"] 是类型标注，告诉开发者和类型检查器：
    这个函数只会返回这两个字符串之一。LangGraph 使用这个映射表
    {"act": "act", "__end__": END} 来将字符串转为实际的目标节点。

    Parameters
    ----------
    state : AgentState
        当前的状态快照。注意：这是 Think 节点执行后的状态。

    Returns
    -------
    Literal["act", "__end__"]
        路由目标："act" 去执行工具，或 "__end__" 终止执行。
    """
    # 情况 1：LLM 已给出最终答案 → 结束
    if state.get("final_answer") is not None:
        return "__end__"

    # 情况 2：LLM 请求调用工具 → 去 Act 节点执行
    if state.get("tool_calls"):
        return "act"

    # 情况 3：兜底 — 无事可做 → 结束
    return "__end__"


# ---------------------------------------------------------------------------
# 图构建函数 — 组装 Agent 的执行流程
# ---------------------------------------------------------------------------

def build_graph(
    tool_registry=None,
    skill_registry=None,
    checkpoint_saver=None,
):
    """
    构建 Agent 的 LangGraph StateGraph。

    这是 Agent 的"组装工厂"：把各个节点和边拼成完整的执行图。

    === 构建步骤详解 ===

    第 1 步：StateGraph(AgentState)
      创建一个空的有向图，并指定状态类型为 AgentState。
      这意味着图中所有节点的输入和输出都必须符合 AgentState 的结构。

    第 2 步：add_node("name", function)
      向图中添加处理节点。每个节点有一个名称和一个异步函数。
      节点函数签名标准：async def node(state: AgentState, config: dict) -> dict
      接收当前状态，返回一个 partial dict（只包含需要更新的字段）。

    第 3 步：set_entry_point("observe")
      设置图的入口节点。每次图开始执行时，从 "observe" 节点开始。
      类比：程序从 main() 开始执行，图从 entry_point 开始执行。

    第 4 步：add_edge("observe", "think")
      添加固定边（普通边）。
      观察完之后，无条件地进入思考。
      这是最简单的连接方式 — A 之后永远是 B。

    第 5 步：add_conditional_edges("think", route_after_think, mapping)
      添加条件边 — LangGraph 最强大的特性之一。
      在 Think 节点执行完后，调用 route_after_think 函数，
      根据返回的字符串和 mapping 字典决定下一步：
        - "act" → 去 act 节点（执行工具）
        - "__end__" → 去 END（LangGraph 的特殊节点，表示图执行终止）

      这相当于程序中的 if-else 分支，但应用于图的拓扑结构。

    第 6 步：add_edge("act", "observe")
      添加循环边 — 这是实现 Agent 循环的关键！
      Act 执行完工具后，回到 Observe 节点，重新准备上下文。
      然后 Observe → Think → ... 形成循环，直到 Think 决定终止。

      如果没有这条边，Agent 只能执行一轮工具调用就得结束，
      那就退化成了普通的单步函数调用。

    第 7 步：builder.compile(checkpointer=...)
      编译图，生成可执行的 Runnable 对象。
      编译过程会：
        - 验证图的拓扑结构是否完整（每个边都指向存在的节点）
        - 将 checkpointer 注册到运行环境中
        - 返回一个可被 .ainvoke() 或 .astream() 调用的对象

    === 参数说明 ===

    Parameters
    ----------
    tool_registry : ToolRegistry, optional
        工具注册表，包含所有可用工具的定义和执行逻辑。
        通过 config["configurable"]["tool_registry"] 传递给各节点。

    skill_registry : SkillRegistry, optional
        技能注册表，包含所有可用技能的定义。
        技能是一组工具的"套餐"，可以限制工具的使用范围。

    checkpoint_saver : BaseCheckpointSaver, optional
        检查点保存器。如果提供，编译时会启用持久化，
        每个 session_id 的状态会被保存，支持多轮对话。
        常用值：
        - MemorySaver() 用于开发测试
        - SqliteSaver(sqlite_conn) 用于生产环境

    Returns
    -------
    CompiledStateGraph
        编译后的可执行图对象。调用方使用：
        - graph.ainvoke(input, config)  执行一次完整的图运行
        - graph.astream(input, config)  流式获取每个节点的输出
    """
    # ----------------------------------------------------------------
    # Step 1: 创建 StateGraph，指定状态类型为 AgentState
    # 这就像声明"这个图的节点都读写 AgentState 结构"
    # ----------------------------------------------------------------
    builder = StateGraph(AgentState)

    # ----------------------------------------------------------------
    # Step 2: 添加三个核心节点
    # 每个节点是一个 async 函数，接收 (state, config)，返回 partial state dict
    # 节点不是在这执行的，只是注册到图中。执行时由 LangGraph 调度。
    # ----------------------------------------------------------------
    builder.add_node("observe", observe_node)  # 观察：准备上下文
    builder.add_node("think", think_node)      # 思考：调用 LLM
    builder.add_node("act", act_node)          # 行动：执行工具

    # ----------------------------------------------------------------
    # Step 3: 设置入口节点
    # 每次图被调用（用户发来新消息），都从 observe 开始
    # ----------------------------------------------------------------
    builder.set_entry_point("observe")

    # ----------------------------------------------------------------
    # Step 4: 固定边 — observe → think
    # 观察完成后，无条件进入思考
    # ----------------------------------------------------------------
    builder.add_edge("observe", "think")

    # ----------------------------------------------------------------
    # Step 5: 条件边 — think → act 或 think → END
    # 根据 LLM 的决策来决定下一步：
    #   - 需要工具？→ 去 act 执行
    #   - 有了最终答案？→ 结束
    #
    # 第三个参数是映射表：将 route_after_think 的返回值映射到图节点
    # "act" 映射到 act 节点，"__end__" 映射到内置的 END 哨兵
    # ----------------------------------------------------------------
    builder.add_conditional_edges(
        "think",                              # 从 think 节点出发
        route_after_think,                    # 使用这个函数决定去向
        {"act": "act", "__end__": END},       # 函数返回值 → 目标节点映射
    )

    # ----------------------------------------------------------------
    # Step 6: 循环边 — act → observe（回到起点！）
    # 这是实现 Agent 多轮工具调用能力的关键。
    #
    # 执行流程：
    #   observe → think → act → observe → think → act → observe → ...
    #                                    ↑___________________________|
    #                                    这个箭头就是 add_edge("act", "observe")
    #
    # 循环终止条件在 think → conditional_edge 中：
    #   final_answer 非空 → END（跳出循环）
    # ----------------------------------------------------------------
    builder.add_edge("act", "observe")

    # ----------------------------------------------------------------
    # Step 7: 编译图
    # checkpointer 提供持久化 — 使 Agent 能记住多轮对话
    #
    # 没有 checkpointer 时：
    #   每轮对话都是独立的，Agent 不记得以前说过什么
    #   适合单次问答
    #
    # 有 checkpointer 时：
    #   同一个 thread_id 的状态会被持久化
    #   第二轮的 messages 自动包含第一轮的所有消息
    #   适合多轮聊天
    # ----------------------------------------------------------------
    if checkpoint_saver:
        return builder.compile(checkpointer=checkpoint_saver)
    return builder.compile()

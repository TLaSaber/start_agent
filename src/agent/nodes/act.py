"""
Act 节点 — Agent 的"双手"
===========================

这个节点负责执行 LLM 在 Think 阶段决定要调用的工具，
并将执行结果注入对话历史。

====================================
Act 节点的核心职责
====================================

1. 接收 tool_calls 列表（LLM 决定要执行的操作）
2. 逐个执行工具调用
3. 将每个工具的执行结果包装为 ToolMessage
4. 返回 ToolMessage 列表和更新后的 loop_count

如果说 Think 是"我想查数据库"，
那么 Act 就是"打开数据库客户端，执行 SQL，拿到结果"。

====================================
ToolMessage 是什么？
====================================

ToolMessage 是 LangChain 中专门用来表示"工具执行结果"的消息类型。

它和普通消息的区别：
  - HumanMessage: 用户说的话
  - AIMessage: AI 说的话（可能带 tool_calls）
  - SystemMessage: 系统指令
  - ToolMessage: 工具执行的结果 —— 这是 LLM 的"手"在外部世界
                 操作后带回来的反馈

ToolMessage 的关键属性：
  - content: 工具执行的结果文本（成功时）或错误信息（失败时）
  - tool_call_id: 对应的 tool_call 的 ID，LangChain 用它把
                  结果和请求匹配起来

为什么需要 ToolMessage？

  想象一次对话：
    User: "今天天气怎么样？"
    AI (AIMessage): tool_calls=[{"name": "get_weather", "id": "call_1"}]
    Tool (ToolMessage): "北京今天晴，25°C", tool_call_id="call_1"
    AI (AIMessage): "北京今天天气晴朗，气温25°C，适合出门。"

  ToolMessage 是 AI 的两句话之间的"桥梁"：
  AI 说"我要查天气" → 工具返回"天气是这样的" → AI 说"好的，我告诉你"

  没有 ToolMessage，AI 就无法知道工具执行了什么、结果是什么，
  也就无法给出基于真实数据的回答。

====================================
权限检查机制
====================================

不是所有工具都能随便执行的。比如"删除数据库"这种操作，
需要明确的权限控制。本系统使用风险等级（risk_level）来控制：

  风险等级分级：
  - low: 只读操作，如查询、读取文件（默认允许）
  - medium: 有限的写操作，如创建文件、修改配置（默认允许）
  - high: 重要写操作，如删除数据、修改系统设置（默认阻止）
  - critical: 危险操作，如格式化磁盘、执行任意命令（默认阻止）

  check_permission 函数就是这道"防火墙"：
    if risk_level in {"high", "critical"}:
      → 拒绝执行，返回拒绝原因

  为什么 high/critical 要被阻止？
    - 安全第一：大模型可能出错或被误导，不能让 AI 随意执行危险操作
    - 最小权限原则：只给 Agent 完成任务所需的最小权限
    - 可审计：被阻止的操作会记录在 ToolMessage 中，用户可以了解

  如果需要允许高风险操作，有两种方式：
    1. 修改 BLOCKED_RISK_LEVELS 集合
    2. 在工具注册时降低风险等级

====================================
Skill 工具白名单
====================================

当用户激活了某个技能时，Act 节点会进行额外的安全检查：

  active_skill.tools 列表定义了该技能允许使用的工具。
  如果 LLM 请求使用不在白名单中的工具，Act 会拒绝执行，
  并返回一条提示信息。

  为什么需要这个？
    - 隔离性：每个技能是一个"沙盒"，只能使用自己声明的工具
    - 防止越权：即使 LLM "想到"要调用其他工具，也执行不了
    - 用户体验：LLM 看到拒绝信息后，会引导用户选择正确的工具

  示例：
    激活了 mysql 技能，但 LLM 想调用 read_file 工具。
    → Act 拒绝："技能 'mysql' 不允许使用工具 'read_file'。
      允许的工具: execute_sql, list_tables, describe_table"

====================================
为什么最后要清空 tool_calls 并增加 loop_count？

  tool_calls = []          ← 清空：本轮工具调用已全部执行完毕
  loop_count = loop_count + 1  ← 递增：记录完成了一轮循环

  清空 tool_calls：如果不清理，下次进入 Think 时，
  route_after_think 会看到还有 tool_calls 没执行，
  就直接又进入 Act，形成死循环。

  递增 loop_count：这是循环计数的唯一递增点。
  每次 Act 完成后 +1，作为 MAX_LOOPS 保护的依据。
"""

from langchain_core.messages import ToolMessage
from src.agent.state import AgentState
from config.settings import MAX_LOOPS


# ---------------------------------------------------------------------------
# 权限白名单
# ---------------------------------------------------------------------------

# 默认允许的风险等级
ALLOWED_RISK_LEVELS = {"low", "medium"}

# 默认阻止的风险等级
BLOCKED_RISK_LEVELS = {"high", "critical"}


def check_permission(tool_name: str, risk_level: str) -> tuple[bool, str]:
    """
    检查某个工具是否被允许执行。返回 (是否允许, 拒绝原因)。

    权限模型：
    - low / medium 风险：自动允许（如读文件、查数据库）
    - high 风险：阻止，提示用户"当前版本暂不支持"
    - critical 风险：阻止，提示用户"特危操作，禁止执行"

    设计理念：
      不是简单地"有权限/无权限"的二元判断，而是区分不同的风险等级
      给出不同的错误提示。这让用户知道为什么操作被阻止，以及
      可以如何调整（比如选择替代工具）。

    Parameters
    ----------
    tool_name : str
        工具名称，用于错误提示中指明哪个工具被阻止

    risk_level : str
        工具的风险等级，取值：low, medium, high, critical

    Returns
    -------
    tuple[bool, str]
        (是否允许, 拒绝原因)。
        允许时原因为空字符串；拒绝时原因为中文提示信息。
    """
    # 如果风险等级在阻止列表中
    if risk_level in BLOCKED_RISK_LEVELS:
        if risk_level == "critical":
            return False, f"工具 '{tool_name}' 为特危操作，禁止执行"
        # risk_level == "high"
        return False, f"工具 '{tool_name}' 为高风险操作，当前版本暂不支持。请使用其他替代工具"

    # 风险等级在 ALLOWED_RISK_LEVELS 中，允许执行
    return True, ""


# ---------------------------------------------------------------------------
# Act 节点主函数
# ---------------------------------------------------------------------------

async def act_node(state: AgentState, config: dict = None) -> dict:
    """
    Act 节点：执行 LLM 请求的工具调用，将结果转换为 ToolMessage。

    这是 Agent 的"行动"阶段。当 Think 节点决定需要调用工具时，
    Act 节点负责实际执行这些工具，并将结果反馈给对话。

    === 处理流程 ===

    ┌────────────────────────────────────────────────────┐
    │  对 tool_calls 中的每个 tool_call:                 │
    │    1. 检查技能白名单（如果技能已激活）              │
    │       ├─ 不在白名单 → 返回 ToolMessage(拒绝)       │
    │       └─ 在白名单   → 继续                        │
    │    2. 查找工具（从 tool_registry）                  │
    │       ├─ 未注册 → 返回 ToolMessage(错误)           │
    │       └─ 已注册 → 继续                            │
    │    3. 权限检查（check_permission）                  │
    │       ├─ 被阻止 → 返回 ToolMessage(权限拒绝)       │
    │       └─ 允许   → 继续                            │
    │    4. 执行工具（tool_registry.execute）              │
    │       ├─ 成功 → 返回 ToolMessage(执行结果)         │
    │       └─ 失败 → 返回 ToolMessage(错误信息)         │
    │  返回所有 ToolMessage + 更新后的 loop_count        │
    └────────────────────────────────────────────────────┘

    Parameters
    ----------
    state : AgentState
        当前 Agent 状态。关键字段：
        - tool_calls: 待执行的工具调用列表
        - loop_count: 当前循环计数
        - active_skill: 当前激活的技能（如果有）

    config : dict, optional
        LangGraph 的 config，通过 config["configurable"] 获取：
        - tool_registry: 工具注册表
        - skill_registry: 技能注册表

    Returns
    -------
    dict
        状态更新字典：
        - messages: ToolMessage 列表（工具执行结果）
        - loop_count: 递增后的循环计数
        - tool_calls: 空列表（清空本轮调用）
    """
    # 获取本轮待执行的工具调用列表
    tool_calls = state.get("tool_calls", [])

    # messages 列表收集所有 ToolMessage（每个工具调用生成一个）
    messages: list[ToolMessage] = []

    # 当前循环计数
    loop_count = state.get("loop_count", 0)

    # 从 config 获取外部依赖
    tool_registry = config.get("configurable", {}).get("tool_registry") if config else None
    skill_registry = config.get("configurable", {}).get("skill_registry") if config else None
    active_skill = state.get("active_skill")

    # ------------------------------------------------------------------
    # 逐个处理 tool_calls 中的每个工具调用
    #
    # 注意：LLM 可能在一次回复中请求调用多个工具。
    # 例如："查天气 + 查新闻" 会一次性发出两个 tool_call，
    # act_node 会全部执行它们，收集所有结果后一起返回。
    # ------------------------------------------------------------------
    for tc in tool_calls:
        tool_name = tc.get("name", "")    # 工具名称，如 "execute_sql"
        tool_args = tc.get("args", {})    # 工具参数，如 {"query": "SELECT ..."}
        call_id = tc.get("id", "")         # 调用 ID，用于匹配 ToolMessage

        # --------------------------------------------------------------
        # 检查 1: 技能工具白名单
        #
        # 如果用户激活了某个技能（如 mysql 技能），
        # 只允许使用该技能在 tools 列表中声明的工具。
        # 这防止 LLM "越狱" — 在 mysql 技能中调用文件系统工具。
        # --------------------------------------------------------------
        if active_skill and skill_registry:
            # 获取完整的技能定义
            skill = skill_registry.get(active_skill.get("name", ""))
            if skill and tool_name not in skill.tools:
                # 工具不在技能白名单中 → 拒绝
                messages.append(ToolMessage(
                    content=f"技能 '{skill.name}' 不允许使用工具 '{tool_name}'。允许的工具: {', '.join(skill.tools)}",
                    tool_call_id=call_id,
                ))
                continue  # 跳过这个调用，处理下一个

        # --------------------------------------------------------------
        # 检查 2: 工具是否存在（是否已注册）
        # --------------------------------------------------------------
        if tool_registry:
            tool = tool_registry.get(tool_name)
            if tool is None:
                # 工具未注册 → 返回错误
                messages.append(ToolMessage(
                    content=f"错误：工具 '{tool_name}' 未注册",
                    tool_call_id=call_id,
                ))
                continue

            # --------------------------------------------------------------
            # 检查 3: 权限检查
            #
            # 根据工具的风险等级（tool.risk_level）判断是否允许执行。
            # high/critical 等级的操作会被阻止。
            # --------------------------------------------------------------
            allowed, reason = check_permission(tool_name, tool.risk_level)
            if not allowed:
                # 权限拒绝 → 返回拒绝原因
                messages.append(ToolMessage(
                    content=f"权限拒绝：{reason}",
                    tool_call_id=call_id,
                ))
                continue

            # --------------------------------------------------------------
            # 检查 4: 执行工具
            #
            # tool_registry.execute(name, **args) 实际执行工具。
            # 返回一个结果对象，包含 .success（是否成功）、
            # .output（成功时的输出）和 .error（失败时的错误信息）。
            #
            # **tool_args 是 Python 的解包语法：
            #   如果 tool_args = {"query": "SELECT 1", "timeout": 5}
            #   则 execute("execute_sql", query="SELECT 1", timeout=5)
            # --------------------------------------------------------------
            result = await tool_registry.execute(tool_name, **tool_args)

            # 根据执行结果构建 ToolMessage 的内容
            content = result.output if result.success else f"执行失败: {result.error}"
        else:
            # 没有 ToolRegistry（开发/测试模式）→ 返回占位信息
            content = f"[ToolRegistry not available] Would execute: {tool_name}({tool_args})"

        # 将工具执行结果包装为 ToolMessage 并加入返回列表
        # ToolMessage 是 LangChain 的标准消息类型，LLM 可以理解它
        messages.append(ToolMessage(content=content, tool_call_id=call_id))

    # ------------------------------------------------------------------
    # 返回状态更新
    #
    # messages: 包含所有 ToolMessage 的列表。
    #   由于 AgentState.messages 使用了 add_messages reducer，
    #   这些 ToolMessage 会被自动追加到消息列表末尾。
    #
    # loop_count: 递增 1，记录完成了一轮循环。
    #   这是循环计数器的唯一递增点 — 每完成一次 Act，计数器 +1。
    #
    # tool_calls: 设为空列表 []，清空本轮的工具调用。
    #   如果不清理，route_after_think 看到还有 tool_calls，
    #   就会再次进入 Act，形成死循环。
    # ------------------------------------------------------------------
    return {
        "messages": messages,            # ToolMessage 列表（自动追加到消息历史）
        "loop_count": loop_count + 1,    # 循环计数 +1
        "tool_calls": [],                # 清空本轮调用（防止死循环）
    }

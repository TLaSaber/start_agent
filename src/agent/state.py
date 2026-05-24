"""
Agent 状态定义模块
===================

这个模块定义了 Agent 在整个对话生命周期中的"状态"数据结构。
在 LangGraph 框架中，State（状态）是最核心的概念 — 它就像一条流水线，
每个节点处理它，然后把它传递给下一个节点。

LangGraph 的核心理念：
  Agent 的执行过程 = 一个有向图上的状态流转
  你定义一个 State 类型，然后图中每个节点读 State / 写 State，
  LangGraph 自动管理状态在节点间的传递。

关键概念讲解：
-------------

1. TypedDict 是什么？
   TypedDict 是 Python 3.8+ 引入的类型提示工具（来自 typing 模块）。
   它让你可以定义一个字典的"结构" — 有哪些键，每个键的值是什么类型。
   对于 LangGraph 来说，TypedDict 就是状态的"蓝图"：
   - 每个键是一个状态字段
   - 每个字段的类型告诉 LangGraph 如何处理数据

2. Annotated[list, add_messages] — LangGraph 的 Reducer 机制
   这是 LangGraph 中最重要也最容易困惑的语法，我们来拆解：

   Annotated[X, Y] 是 Python 的类型标注工具，意思是：
     "类型是 X，但用 Y 这个函数来处理它"

   在 LangGraph 中：
     Annotated[list[BaseMessage], add_messages]
     类型: list[BaseMessage]（消息列表）
     Reducer: add_messages（消息合并函数）

   什么是 Reducer？Reducer 决定了当多个节点都对同一个字段写入时，
   "新值"和"旧值"如何合并。

   普通字段（没有 Annotated）的行为是：新值覆盖旧值。
   比如 loop_count: int — 每次写新值，旧值就没了。

   但 messages 字段用了 add_messages reducer：
   当你 return {"messages": [new_msg]} 时，add_messages 不会覆盖旧消息，
   而是把 new_msg 追加到已有消息列表末尾。

   这就是 LangGraph 实现"对话历史自动累积"的魔法！
   你永远不需要手动拼接消息列表，只需返回新消息，框架帮你追加。

   这也是为什么 add_messages 是 LangGraph 内置的 reducer —
   它从 langgraph.graph.message 导入，专门为消息列表设计。

3. 这个 State 的设计理念：
   这个 AgentState 承载了 Agent 在 Observe → Think → Act 循环中
   需要保留的所有信息。每次循环，各个节点读取和修改这个状态，
   状态在节点间流转，直到达到终止条件。
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# ---------------------------------------------------------------------------
# ToolCall — LLM 请求调用工具的标准化格式
# ---------------------------------------------------------------------------

class ToolCall(TypedDict):
    """
    LLM 发起的工具调用请求的数据结构。

    当 LLM（大语言模型）判断需要用工具来完成用户的请求时，
    它会在 AIMessage 中附带 tool_calls 字段。
    这个 TypedDict 定义了每个 tool_call 条目的格式。

    这实际上是 OpenAI Function Calling 协议的标准格式，
    LangChain/LangGraph 在此基础上做了封装。

    字段说明：
        name (str): 要调用的工具名称，例如 "web_search"、"read_file"
        args (dict): 传递给工具的参数，键值对形式的字典
                     例如 {"query": "今天天气", "location": "北京"}
        id (str): 工具调用的唯一标识符，由 LLM 生成
                  用于后续将执行结果（ToolMessage）与调用请求配对
    """
    name: str
    args: dict
    id: str


# ---------------------------------------------------------------------------
# AgentState — Agent 的全局对话状态
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Agent 的全局状态定义 — LangGraph 图中的"共享记忆"。

    这个字典在图的每个节点（Observe / Think / Act）之间流转。
    每个节点可以读取当前状态，也可以返回一个 partial dict 来更新部分字段。

    ----------------------------------------
    字段详解：
    ----------------------------------------

    messages: Annotated[list[BaseMessage], add_messages]
        对话消息列表，是整个 Agent 的"记忆中枢"。

        它包含所有历史消息：SystemMessage（系统指令）、
        HumanMessage（用户输入）、AIMessage（AI 回复）、
        ToolMessage（工具执行结果）。

        【关键】使用了 add_messages reducer：
        每个节点返回 {"messages": [新消息]} 时，
        新消息会被追加到已有列表末尾，而不是覆盖。
        这意味着对话历史会自动累积，无需手动管理。

        类比：就像微信群聊记录，每个人说一句就自动追加到聊天记录中，
        你不需要每次发言前先收集之前的聊天记录再拼上。

    session_id: str
        会话的唯一标识符。用于区分不同的对话会话。

        实际用途：
        - LangGraph 的 checkpointer 用它来区分不同会话的检查点
        - 多用户场景下，每个用户可以有多个 session
        - 前端可以用它来恢复历史对话

    user_id: str
        用户的唯一标识符。用于区分不同的用户。
        同一个用户可以有多轮对话（多个 session），
        但 user_id 保持不变，方便做用户级别的记忆管理。

    recalled_memories: list[dict]
        从记忆系统中召回的、与当前对话相关的记忆片段。

        每个元素是一个 dict，通常包含：
        - content: 记忆内容
        - category: 记忆分类（fact, preference, event 等）
        - source: 记忆来源

        这些记忆在 Observe 节点中被注入到 System Prompt，
        帮助 LLM "记住"用户的偏好和历史信息。

    active_skill: dict | None
        当前激活的技能。为 None 表示没有技能被激活。

        当用户的消息触发了某个技能时（比如用户说"用 mysql 查询"），
        Think 节点会设置这个字段。激活的技能包含：
        - name: 技能名称
        - description: 技能描述和指令
        - tools: 该技能允许使用的工具白名单

        在 Act 节点中，如果技能被激活，会进行工具白名单检查 —
        只允许执行该技能声明过的工具，提供安全边界。

    tool_calls: list[dict]
        当前轮次待执行的工具调用列表。

        这是 LLM 在 Think 节点中决定要调用的工具。每个元素包含：
        - name: 工具名
        - args: 工具参数
        - id: 调用 ID（与 ToolCall 对应）

        Think 节点写入这些调用，Act 节点读取并执行它们，
        执行完后清空此列表（设为 []）。

        如果这个列表为空且 final_answer 也为空，
        说明 LLM 没有需要调用的工具，准备结束对话。

    loop_count: int
        Agent 循环计数器。记录已经完成的 Observe → Think → Act 循环次数。

        这是一个安全机制：每完成一次 Act，计数器 +1。
        当 loop_count 达到 MAX_LOOPS（默认 15）时，
        Think 节点会强制终止，防止 Agent 陷入无限循环。

        类比：就像微波炉的定时器，设定最长加热时间，
        防止忘了关而一直转。

    final_answer: str | None
        Agent 的最终回复。为 None 表示对话尚未结束。

        当 LLM 决定直接回答用户（而不是调用工具）时，
        Think 节点会设置这个字段为具体的回复文本。

        它是图执行终止的信号：route_after_think 检查此字段，
        如果非 None，则路由到 END 终止执行。

        最终，这个字段的内容会被返回给用户。

    compact_summary: str | None
        对话历史的压缩摘要。为 None 表示尚未进行压缩。

        当对话历史过长，达到 token 阈值时，
        Observe 节点会将早期消息压缩为一段摘要文本，
        并在 System Prompt 前面加上"## 历史对话摘要"。

        这样可以让 LLM 了解之前的对话要点，同时不超出 token 限制。
        None 表示尚不需要压缩（对话还很短）。
    """
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_id: str
    recalled_memories: list[dict]
    active_skill: dict | None
    tool_calls: list[dict]
    loop_count: int
    final_answer: str | None
    compact_summary: str | None

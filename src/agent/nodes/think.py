"""
Think 节点 — Agent 的"大脑"
=============================

这个节点是 Agent 推理的核心：调用大语言模型（LLM），
让它根据当前对话上下文决定下一步做什么。

====================================
Think 节点的核心职责
====================================

1. 调用 LLM：把当前消息列表发给模型
2. 解析 LLM 的回复：区分"直接回答"和"需要调用工具"
3. 检测技能触发：用户说了什么关键词来激活技能
4. 兜底保护：循环次数限制、LLM 重试机制

====================================
parse_llm_response — AI 回复的两种形态
====================================

LLM（比如 GPT-4）在收到消息后，可能返回两种类型的回复：

  类型 A：直接回答（Direct Answer）
    LLM 说："好的，根据你的需求，答案是 42。"
    这时 AIMessage.content 包含回答文本
    AIMessage.tool_calls 为空

  类型 B：工具调用（Function Calling）
    LLM 说："我需要查一下天气API..."
    这时 AIMessage.content 可能为空（或包含解释）
    AIMessage.tool_calls 非空，包含要调用的工具名称和参数

  parse_llm_response 的工作就是检查 AIMessage.tool_calls：
    - 有 tool_calls → 返回 tool_calls 列表（给 Act 节点执行）
    - 无 tool_calls → 返回 final_answer（直接结束对话）

====================================
AIMessage.tool_calls 的结构
====================================

这是 OpenAI Function Calling 协议的标准格式。当一个 AIMessage 包含
tool_calls 时，它的结构是这样的：

  AIMessage(
    content="我需要查询数据库...",
    tool_calls=[
      {
        "name": "execute_sql",        # 工具名称
        "args": {                     # 工具参数
          "query": "SELECT * FROM users WHERE id = 1"
        },
        "id": "call_abc123"           # 调用 ID（用于匹配结果）
      },
      {
        "name": "web_search",         # 可以一次调用多个工具！
        "args": {"query": "Python LangGraph tutorial"},
        "id": "call_def456"
      }
    ]
  )

  每个 tool_call 的含义：
    - name: 要调用的工具名称，对应 tool_registry 中注册的工具
    - args: 传递给工具的参数，JSON 格式
    - id: 唯一标识符。LLM 给每个调用分配一个 ID，
           执行结果（ToolMessage）也用同一个 ID 标记，
           LangChain 通过 ID 将结果与调用匹配

====================================
LLM 重试机制 — 指数退避
====================================

网络请求、LLM API 调用都可能因为各种原因失败：
  - 网络超时
  - API 限流（rate limit）
  - 服务暂时不可用

所以 think_node 使用指数退避重试：

  第 1 次失败 → 等待 2^0 = 1 秒后重试
  第 2 次失败 → 等待 2^1 = 2 秒后重试
  第 3 次失败 → 等待 2^2 = 4 秒后重试

  重试次数由 LLM_MAX_RETRIES（默认 3）控制。
  如果全部失败，返回错误信息作为 final_answer。

为什么用指数退避而不是固定间隔？
  固定间隔：1s, 1s, 1s — 如果 API 短时间内确实不可用，频繁重试无意义
  指数退避：1s, 2s, 4s — 每次等待更久，给服务恢复时间，也降低了请求频率

====================================
MAX_LOOPS — 循环次数兜底保护
====================================

Agent 的 Observe→Think→Act 循环可以执行多轮，但如果不设上限，
可能出现"无限循环"的问题：

  场景：LLM 反复调用同一个工具，每次都得到类似的结果，但就是
  不给出最终答案，一直在"观察→思考→执行→观察..."中打转。

  解决方案：loop_count 计数器。每完成一次 Act，counter +1。
  当 loop_count >= MAX_LOOPS（默认 15）时，强制终止，
  返回"达到最大循环次数"的提示。

  这是 Agent 开发的常见实践：始终给循环设置上限，
  防止 API 调用无限消耗和用户体验恶化。
"""

import asyncio
import re
from langchain_core.messages import AIMessage, HumanMessage
from src.agent.state import AgentState
from config.settings import MAX_LOOPS, LLM_MAX_RETRIES


# ---------------------------------------------------------------------------
# LLM 回复解析
# ---------------------------------------------------------------------------

def parse_llm_response(response: AIMessage) -> dict:
    """
    解析 LLM 的回复，区分"直接回答"和"工具调用请求"。

    这是 Think 节点的核心决策逻辑。LLM 的回复有两种可能：

    1. 直接回答用户问题：
       AIMessage.tool_calls 为空
       → 从 AIMessage.content 提取文本作为 final_answer
       → 这意味着本轮对话可以结束了

    2. 请求调用工具：
       AIMessage.tool_calls 非空（至少有一个 tool_call）
       → 提取所有 tool_call 条目，转为标准格式
       → 这些会传给 Act 节点去执行

    返回的 dict 结构：
      {
        "tool_calls": [{"name": "xxx", "args": {...}, "id": "yyy"}, ...],
        "final_answer": "回复文本" or None
      }

    Parameters
    ----------
    response : AIMessage
        LLM 返回的原始消息对象

    Returns
    -------
    dict
        包含 tool_calls 和 final_answer 的字典。
        两者互斥：有 tool_calls 时 final_answer 为 None，
        反之亦然。
    """
    # 检查 LLM 是否想调用工具
    # tool_calls 是 AIMessage 的属性，当 LLM 决定使用工具时自动填充
    if response.tool_calls and len(response.tool_calls) > 0:
        # 情况 B：LLM 请求调用工具
        # 将每个 tool_call 展开为标准格式的 dict
        return {
            "tool_calls": [
                {
                    "name": tc["name"],           # 工具名称
                    "args": tc["args"],           # 工具参数（dict）
                    "id": tc.get("id", "")         # 调用 ID（用于工具结果匹配）
                }
                for tc in response.tool_calls
            ],
            "final_answer": None,  # 没有最终答案，还需要继续
        }

    # 情况 A：LLM 直接回答（不需要调用工具）
    return {
        "tool_calls": [],                      # 没有工具调用
        "final_answer": (
            response.content                    # AI 的回复文本
            if isinstance(response.content, str)
            else str(response.content)          # 兼容非字符串内容
        ),
    }


# ---------------------------------------------------------------------------
# 记忆归档触发检测
# ---------------------------------------------------------------------------

# ARCHIVE_TRIGGERS 定义了正则表达式列表，用于检测用户消息中是否包含
# "请记住这个" 或 "保存这个偏好" 之类的归档指令。
#
# 每条正则匹配一种中文表达方式：
#   r"记住[，,：:]"  → "记住，"、"记住："、"记住，" 等
#   r"保存[这那]"     → "保存这"、"保存那"
#   r"归档[这那]"     → "归档这"、"归档那"
#   r"记录下[来这]"   → "记录下来"、"记录下这"
#   r"我偏好"        → "我偏好..."（直接匹配偏好表达）
#   r"我习惯"        → "我习惯..."
#   r"我不喜欢"      → "我不喜欢..."
#   r"我总是"        → "我总是..."
#   r"我的.*是"      → "我的名字是"、"我的爱好是" 等
#   r"备忘[，,：:]"  → "备忘，"、"备忘：" 等
#   r"记一下"        → "记一下..."
#   r"别忘了"        → "别忘了..."
#
# 这个模式列表可以根据需要扩展，覆盖更多的中文表达方式。
ARCHIVE_TRIGGERS = [
    r"记住[，,：:]",
    r"保存[这那]",
    r"归档[这那]",
    r"记录下[来这]",
    r"我偏好",
    r"我习惯",
    r"我不喜欢",
    r"我总是",
    r"我的.*是",
    r"备忘[，,：:]",
    r"记一下",
    r"别忘了",
]


def detect_archive_triggers(user_message: str) -> list[dict]:
    """
    检测用户消息中是否包含"请记住这个"之类的记忆归档触发词。

    当用户在对话中说"记住，我喜欢简洁的回答"之类的话时，
    这个函数会检测到并生成一条待归档的记忆条目。

    检测逻辑：
    - 遍历 ARCHIVE_TRIGGERS 中的正则模式
    - 对用户消息做正则匹配（re.search）
    - 一旦匹配成功，生成一条记忆条目并立即返回（break）
    - 只匹配第一个触发词，不重复归档

    生成的记忆条目格式：
    {
      "content": "用户消息原文",
      "source": "user_command",    # 来源标记：用户主动要求的归档
      "category": "preference"     # 类别标记：偏好类信息
    }

    Parameters
    ----------
    user_message : str
        用户输入的原始消息文本

    Returns
    -------
    list[dict]
        匹配到的归档条目列表。如果没有触发词，返回空列表。
        目前最多返回一个元素（匹配到第一个就停止）。
    """
    triggers = []
    for pattern in ARCHIVE_TRIGGERS:
        # re.search 在整个字符串中搜索正则匹配
        # 只要找到任意一处匹配就返回 Match 对象，否则返回 None
        if re.search(pattern, user_message):
            triggers.append({
                "content": user_message.strip(),
                "source": "user_command",
                "category": "preference",
            })
            break  # 只匹配第一个触发词，避免重复
    return triggers


# ---------------------------------------------------------------------------
# Think 节点主函数
# ---------------------------------------------------------------------------

async def think_node(state: AgentState, config: dict = None) -> dict:
    """
    Think 节点：调用 LLM 进行推理，决定下一步行动。

    这是 Agent 的"决策中枢"。接收 Observe 准备好的上下文，
    调用 LLM 思考，然后决定：
    - 直接回答用户（返回 final_answer）
    - 调用工具获取更多信息（返回 tool_calls）

    === 完整的处理流程 ===

    ┌────────────────────────────────────────────────────┐
    │  1. 从 config 获取 LLM 模型实例                    │
    │  2. 检查循环次数（MAX_LOOPS 保护）                 │
    │  3. 调用 LLM.ainvoke(messages)                     │
    │     ├─ 成功 → 解析回复                            │
    │     └─ 失败 → 指数退避重试（最多 LLM_MAX_RETRIES） │
    │  4. 检测技能触发（Fix 4）                          │
    │  5. 返回解析结果                                   │
    └────────────────────────────────────────────────────┘

    Parameters
    ----------
    state : AgentState
        当前 Agent 状态。关键字段：
        - messages: 包含 SystemMessage + 历史对话的消息列表
        - loop_count: 当前循环次数

    config : dict, optional
        LangGraph 的 config 对象，通过 config["configurable"] 获取：
        - model: LLM 模型实例（必需，否则报错）
        - skill_registry: 技能注册表（用于检测技能触发）

    Returns
    -------
    dict
        包含以下部分或全部字段的状态更新：
        - tool_calls: 待执行的工具调用列表
        - final_answer: 最终回复文本
        - active_skill: 触发的新技能定义（如果有）
    """
    # ------------------------------------------------------------------
    # Step 1: 获取 LLM 模型实例
    #
    # 模型通过 config["configurable"]["model"] 传入。
    # 这是 LangGraph 的依赖注入模式 — 不在节点内硬编码模型实例，
    # 这样可以在测试中用 mock 模型，在生产中用真实模型。
    # ------------------------------------------------------------------
    model = config.get("configurable", {}).get("model") if config else None
    if model is None:
        return {"final_answer": "Model not configured", "tool_calls": []}

    # ------------------------------------------------------------------
    # Step 2: MAX_LOOPS 兜底保护
    #
    # 检查是否已经循环太多次了。如果 loop_count 达到 MAX_LOOPS，
    # 强制终止，返回提示信息。这是防止 Agent 陷入无限循环的
    # 安全网 — 类似于电路中的保险丝。
    # ------------------------------------------------------------------
    if state.get("loop_count", 0) >= MAX_LOOPS:
        return {
            "final_answer": "达到最大循环次数，任务中断。已完成的部分已返回。",
            "tool_calls": [],
        }

    # 获取当前消息列表（包含 SystemMessage + 历史对话）
    messages = state.get("messages", [])

    # ------------------------------------------------------------------
    # Step 3: 调用 LLM，带指数退避重试
    #
    # model.ainvoke(messages) 是 LangChain 的异步调用接口。
    # 传入完整的消息列表，LLM 根据所有上下文进行推理。
    #
    # 重试逻辑：
    #   for attempt in range(LLM_MAX_RETRIES):
    #     try → 成功则 break
    #     except → 等待 2^attempt 秒后重试
    #            → 如果是最后一次尝试，返回错误信息
    #
    # 指数退避时间表（LLM_MAX_RETRIES=3）：
    #   attempt=0: 立即尝试，失败后等 1s
    #   attempt=1: 等 1s 后尝试，失败后等 2s
    #   attempt=2: 等 2s 后尝试，失败 → 返回错误
    # ------------------------------------------------------------------
    last_error = None
    response = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            # ainvoke 是 async invoke 的缩写 — 异步调用 LLM
            response = await model.ainvoke(messages)
            break  # 成功，跳出重试循环
        except Exception as e:
            last_error = e
            if attempt < LLM_MAX_RETRIES - 1:
                # 计算退避等待时间：2^attempt 秒
                # attempt=0 → 1s, attempt=1 → 2s, attempt=2 → 4s
                wait = 2 ** attempt  # 1s, 2s, 4s
                await asyncio.sleep(wait)
            else:
                # 最后一次尝试也失败了 → 放弃，返回错误
                return {
                    "final_answer": f"LLM 调用失败（已重试 {LLM_MAX_RETRIES} 次）: {str(last_error)}",
                    "tool_calls": [],
                }

    # ------------------------------------------------------------------
    # Step 4: 解析 LLM 回复
    #
    # parse_llm_response 会判断 LLM 是给出了最终答案还是需要调用工具。
    # 返回的 parsed dict 包含 tool_calls 和/或 final_answer。
    # ------------------------------------------------------------------
    parsed = parse_llm_response(response)
    new_state = dict(parsed)

    # ------------------------------------------------------------------
    # Step 5: 技能渐进发现（Fix 4）
    #
    # 检查用户的最后一条消息是否提到了某个技能的名称。
    # 如果匹配，自动激活该技能，设置 active_skill 字段。
    #
    # 为什么在这里做？因为技能激活可能需要 LLM 的推理结果，
    # 也作为一个安全层 — 即使用户没有显式声明技能，
    # 系统也可以通过关键词匹配自动激活。
    #
    # 流程：
    #   1. 从后往前找最后一条 HumanMessage（用户输入）
    #   2. 遍历所有已注册技能
    #   3. 如果技能名称出现在用户消息中（不区分大小写）
    #   4. 激活该技能 → 设置 active_skill 包含 name, description, tools
    #
    # 示例：用户输入 "用 mysql 查一下用户表"
    #   → skill_summary["name"] = "mysql"
    #   → "mysql" in "用 mysql 查一下用户表" → True
    #   → 激活 mysql 技能，限制只能使用 MySQL 相关工具
    # ------------------------------------------------------------------
    skill_registry = config.get("configurable", {}).get("skill_registry") if config else None
    if skill_registry:
        last_human = None
        # 从消息列表末尾开始反向遍历，找最近一条用户消息
        # reversed() 从最新的消息开始找，效率更高
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        if last_human:
            # 遍历所有注册的技能，检查名称是否出现在用户消息中
            for skill_summary in skill_registry.get_summaries():
                # 简单的子串匹配（不区分大小写）
                if skill_summary["name"] in last_human.lower():
                    # 命中！获取完整技能定义并激活
                    skill = skill_registry.get(skill_summary["name"])
                    if skill:
                        new_state["active_skill"] = {
                            "name": skill.name,
                            "description": skill.description,
                            "tools": skill.tools,       # 工具白名单！
                        }
                        break  # 只激活一个技能

    return new_state

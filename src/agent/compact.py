"""
上下文压缩模块（compact.py）
===========================

这个模块负责在对话历史过长时，将早期消息压缩为一段摘要，
防止超出 LLM 的 token 限制。

====================================
为什么需要上下文压缩？
====================================

大语言模型（LLM）有一个硬性的限制：上下文窗口（Context Window）。

上下文窗口是 LLM 一次能处理的 token 上限。不同模型有不同的窗口大小：
  - GPT-4: 8K 或 128K token
  - GPT-3.5: 4K 或 16K token
  - Claude: 200K token

当对话历史中的 token 数接近或超过这个限制时，会出现问题：
  1. API 直接拒绝请求（token 超限）
  2. LLM 的推理质量下降（"注意力"被稀释）
  3. API 费用飙升（每次请求都带着越来越长的历史）

所以在多轮对话中，我们需要一种机制来"压缩"早期消息。

====================================
压缩策略
====================================

本模块采用的策略是"保留近期 + 摘要远古"：

  完整消息列表：
  [msg1, msg2, msg3, msg4, msg5, msg6, msg7, msg8, msg9, msg10]
   │                                        │                    │
   └────── 早期消息（被压缩为摘要）──────┘  └── 近期消息（保留原文）──┘

  压缩后变成：
  [摘要文本（放在 System Prompt 中）, msg5, msg6, msg7, msg8, msg9, msg10]

  为什么保留最近的 N 条消息？
  - 最近的对话上下文最重要，LLM 需要原文来理解当前意图
  - 早期对话可以概括为要点（"用户说想要简洁回答"）
  - 这是一种平衡：节省 token 的同时，尽量不丢失信息

====================================
estimate_tokens 的原理
====================================

Token（词元）是 LLM 处理文本的最小单位。它不是按"字"或"词"来算的，
而是由模型的分词器（tokenizer）来切分。

  tiktoken 是 OpenAI 开源的分词库，可以精确计算一段文本的 token 数。

  示例（GPT-4 的分词）：
    "Hello world"  → 2 tokens （"Hello", " world"）
    "你好世界"      → 3 tokens （"你", "好", "世界"）
    "ChatGPT is great" → 5 tokens

  为什么不能按字数估算？
  - 不同语言的 token 密度不同：英文 1 词 ≈ 1.3 tokens，中文 1 字 ≈ 1.5-2 tokens
  - 代码、数字、标点符号的 token 化规则也不同
  - 只有用 tokenizer 才能精确计算

  cl100k_base 是什么？
  - GPT-4 和 GPT-3.5-turbo 使用的编码器名称
  - 如果模型名称无法识别（如自定义模型），回退到这个通用编码器
"""

import tiktoken
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Token 估算
# ---------------------------------------------------------------------------

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """
    使用 tiktoken 精确估算文本的 token 数量。

    Token 不是字符，不是字节，而是 LLM 分词后的最小语义单元。
    只有使用与模型匹配的编码器才能准确计算。

    工作流程：
      1. 尝试获取指定模型的编码器
         - encoding_for_model("gpt-4") → 返回 GPT-4 的分词器
      2. 如果模型名称未知，回退到 cl100k_base 编码器
         - cl100k_base 是 GPT-4/3.5 系列使用的通用编码
      3. 用编码器对文本进行 encode（分词），返回 token 列表
      4. 返回 token 列表的长度

    为什么用 tiktoken 而不是简单的 len(text)？
      中英文混合文本中，简单字符计数偏差巨大：
        "你好" (2个字符) ≈ 2-3 tokens
        "Hello" (5个字符) ≈ 1 token
      所以必须用 tokenizer 精确计算。

    Parameters
    ----------
    text : str
        要估算的文本内容

    model : str, optional
        模型名称，默认 "gpt-4"。用于选择正确的分词器。

    Returns
    -------
    int
        文本的 token 数量
    """
    try:
        # 尝试获取指定模型的编码器
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # 模型名称未知时回退到 cl100k_base 编码器
        # cl100k_base 是 GPT-4, GPT-3.5-turbo, text-embedding-ada-002 使用的编码
        encoding = tiktoken.get_encoding("cl100k_base")
    # encode 将文本转为 token ID 列表，len() 返回 token 数量
    return len(encoding.encode(text))


def estimate_messages_tokens(messages: list[BaseMessage]) -> int:
    """
    估算整个消息列表的总 token 数量。

    这个函数遍历消息列表中的每一条消息，提取其 content 文本，
    然后用 estimate_tokens 计算 token 数，最后求和。

    注意：这里只计算了消息内容的 token 数，不包括：
    - 消息类型标记的额外开销（如 role: "user"）
    - 每条消息之间的分隔符
    - 这些开销通常占比较小（每条约 3-5 tokens），可以忽略

    Parameters
    ----------
    messages : list[BaseMessage]
        消息列表，包含 SystemMessage, HumanMessage, AIMessage, ToolMessage 等

    Returns
    -------
    int
        所有消息内容的总 token 数
    """
    total = 0
    for msg in messages:
        # 提取消息文本内容
        # content 通常是 str，但有些 LangChain 消息类型可能返回 list
        # 为了兼容性，非 str 时转为 str
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += estimate_tokens(content)
    return total


# ---------------------------------------------------------------------------
# 压缩判断
# ---------------------------------------------------------------------------

def should_compact(messages: list[BaseMessage], threshold_ratio: float, max_tokens: int) -> bool:
    """
    判断当前消息列表是否需要压缩。

    判断逻辑：
      1. 如果消息列表为空 → 不需要压缩
      2. 估算当前消息的总 token 数
      3. 计算阈值 = max_tokens * threshold_ratio
      4. 如果估算值 > 阈值 → 需要压缩

    为什么用 threshold_ratio 而不是直接用 max_tokens？
      如果等 token 数达到 max_tokens 才压缩，可能已经太晚了 —
      下一轮对话再加几条消息就超限了。使用 80% 的阈值（threshold_ratio=0.8）
      可以提前压缩，给后续消息留出空间。

    示例计算：
      max_tokens = 65536 (GPT-4 的 64K 上下文)
      threshold_ratio = 0.8
      阈值 = 65536 * 0.8 = 52428 tokens
      当消息超过 52428 tokens 时触发压缩

    Parameters
    ----------
    messages : list[BaseMessage]
        消息列表

    threshold_ratio : float
        阈值比例，取值范围 0.0-1.0。默认 0.8 表示 80% 时触发压缩。

    max_tokens : int
        模型的最大上下文窗口大小

    Returns
    -------
    bool
        True 表示需要压缩，False 表示不需要
    """
    if not messages:
        return False
    # 估算当前总 token 数
    estimated = estimate_messages_tokens(messages)
    # 计算压缩阈值
    threshold = int(max_tokens * threshold_ratio)
    # 超过阈值就压缩
    return estimated > threshold


# ---------------------------------------------------------------------------
# 执行压缩
# ---------------------------------------------------------------------------

async def compact_messages(
    messages: list[BaseMessage],
    chat_model,
    keep_recent: int = 6,
) -> tuple[str, list[BaseMessage]]:
    """
    将消息列表的早期部分压缩为一段摘要文本。

    这是压缩的核心函数，执行以下步骤：

    === 压缩流程 ===

    原始消息列表（假设 15 条消息，keep_recent=6）：

      [M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, M13, M14, M15]
       │                                        │                          │
       └─── old_messages (切片 [0:9]) ────────┘  └── recent_messages (切片 [9:15]) ──┘

    1. 计算分界线：boundary = len(messages) - keep_recent
       → 15 - 6 = 9
       → 前 9 条为旧消息，后 6 条为近期消息

    2. 如果消息总数 <= keep_recent → 消息还很少，不需要压缩，直接返回

    3. 将旧消息格式化为一段文本：
       每条消息截取前 500 个字符（避免摘要请求本身就太长）

    4. 调用 LLM（chat_model）生成摘要：
       发送一个"请将以下对话历史压缩为一段简洁的摘要"的请求

    5. 返回：(摘要文本, 近期的完整消息列表)

    === 为什么用 LLM 生成摘要？ ===

    简单的截断（只保留最近 N 条，丢弃旧的）会丢失信息。
    用 LLM 生成摘要是"有损压缩" — 信息量少了，但关键信息被保留了。

    类比：就像会议记录 — 你不会逐字逐句记住整个会议，
    而是记住"讨论了项目进度，决定延后到下周，张三负责UI部分"。

    === 返回值说明 ===

    summary 的使用方式（在 observe_node 中）：
      被放到 System Prompt 的最前面：
        "## 历史对话摘要\n{summary}\n\n## 可用技能\n..."

    recent_messages 的使用方式：
      这 6 条消息保持原样，追加在 SystemMessage 之后，
      LLM 可以直接阅读最近对话的原文。

    Parameters
    ----------
    messages : list[BaseMessage]
        完整的消息列表

    chat_model : BaseChatModel
        LLM 模型实例，用于生成摘要。需要有 ainvoke 方法（异步调用）。

    keep_recent : int, optional
        保留的最近消息数量，默认 6 条。
        这个数字的选择：
        - 太少（如 2-3）：可能丢失重要上下文
        - 太多（如 20+）：压缩效果不明显
        - 6 是一个经验值：保留最近 3 轮对话（用户-AI-用户-AI-用户-AI）

    Returns
    -------
    tuple[str, list[BaseMessage]]
        (摘要文本, 近期消息列表)。
        如果消息数量不足不需要压缩，返回 ("", 原始消息列表)。
    """
    # 如果消息总数不超过保留数量，无需压缩
    if len(messages) <= keep_recent:
        return "", messages

    # 计算分界点：前 boundary 条是旧消息，后面的是近期消息
    boundary = len(messages) - keep_recent

    # 切片取旧消息和近期消息
    # messages[:boundary] — 前 9 条（将被压缩）
    # messages[boundary:] — 后 6 条（保留原文）
    old_messages = messages[:boundary]
    recent_messages = messages[boundary:]

    # ------------------------------------------------------------------
    # 将旧消息格式化为 LLM 可读的文本
    #
    # 每条消息格式： [消息类型]: 内容（截取前 500 字符）
    # 示例输出：
    #   [human]: 帮我查一下今天的新闻
    #   [ai]: 好的，我先搜索一下。tool_calls=[...]
    #   [tool]: 搜索结果：以下是今天的热点新闻...
    #   [ai]: 根据搜索结果，今天的主要新闻有：1. ...
    #
    # 为什么截取 500 字符？
    #   工具返回的结果可能很长（如网页全文），如果全部发给摘要模型，
    #   摘要请求本身就消耗大量 token，得不偿失。
    #   500 字符足够保留关键信息，又限制了请求长度。
    # ------------------------------------------------------------------
    old_text = "\n".join(
        f"[{m.type}]: {m.content if isinstance(m.content, str) else str(m.content)[:500]}"
        for m in old_messages
    )

    # ------------------------------------------------------------------
    # 构造摘要请求并发送给 LLM
    #
    # 使用两条消息：
    #   SystemMessage: 告诉 LLM "你是一个摘要助手"
    #   HumanMessage:  包含待摘要的对话历史
    #
    # 这不是发到用户的对话中，而是一个独立的、仅用于生成摘要的请求。
    # ------------------------------------------------------------------
    summary_prompt = (
        f"请将以下对话历史压缩为一段简洁的摘要"
        f"（保留关键决策、用户偏好和重要信息）：\n\n{old_text}"
    )

    response = await chat_model.ainvoke([
        SystemMessage(content="你是一个对话摘要助手。用中文输出简洁摘要。"),
        HumanMessage(content=summary_prompt),
    ])

    # 提取摘要文本
    summary = response.content if isinstance(response.content, str) else str(response.content)

    # 返回：摘要文本 + 保留的近期消息
    return summary, recent_messages

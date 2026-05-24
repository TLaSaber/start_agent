"""
聊天（Chat）路由 —— POST /chat 对话端点。

本模块是 PyAgent 最核心的 API 端点，负责处理用户发送的每一条消息。
整个聊天请求的执行流程体现了 PyAgent 的"记忆召回 → Agent 循环 →
归档检测"三步处理模式。
"""

import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text
from langchain_core.messages import HumanMessage, AIMessage

from src.server.schemas.request import ChatRequest
from src.server.schemas.response import ChatResponse
from src.db.engine import async_session_factory
from config.settings import MEMORY_RECALL_TOP_K

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_handler(request: ChatRequest, req: Request):
    """处理一次对话请求 —— POST /chat 端点的主入口。

    完整的执行流程包含以下 6 个步骤：

    ┌─────────────────────────────────────────────────────────┐
    │ Step 1: 会话验证 / 创建                                 │
    │ ├─ 如果提供了 session_id → 验证会话存在且未关闭         │
    │ └─ 如果没有 session_id → 创建新会话                     │
    ├─────────────────────────────────────────────────────────┤
    │ Step 2: 长期记忆召回                                    │
    │ └─ 从记忆库中检索与用户消息相关的历史信息               │
    ├─────────────────────────────────────────────────────────┤
    │ Step 3: 构建初始 Agent 状态                             │
    │ ├─ 将用户消息包装为 HumanMessage                        │
    │ ├─ 将工具绑定到模型（function calling）                  │
    │ └─ 组装包含消息、会话信息、记忆、技能等上下文的状态      │
    ├─────────────────────────────────────────────────────────┤
    │ Step 4: 执行 Agent 循环                                  │
    │ ├─ 获取或构建 LangGraph 执行图（缓存）                   │
    │ └─ 使用 ainvoke 异步执行，直到 Agent 产生最终答案       │
    ├─────────────────────────────────────────────────────────┤
    │ Step 5: 更新会话元数据                                   │
    │ └─ message_count +1, 更新 updated_at                     │
    ├─────────────────────────────────────────────────────────┤
    │ Step 6: 归档检测                                        │
    │ └─ 检测用户消息中是否包含需要归档为长期记忆的信息        │
    └─────────────────────────────────────────────────────────┘

    参数：
        request: ChatRequest，包含 message、user_id、可选的 session_id。
        req: FastAPI 的 Request 对象，通过 req.app.state 访问全局状态。

    返回：
        ChatResponse，包含 session_id、answer（AI 的回答）、
        以及 loop_count（Agent 循环执行的步数）。
    """
    # 从应用状态中获取各个全局组件
    app_state = req.app.state
    tool_registry = app_state.tool_registry
    skill_registry = app_state.skill_registry
    model_provider = app_state.model_provider
    # memory_provider 使用 getattr 安全访问，可能在极早期启动失败时不存在
    memory_provider = getattr(app_state, "memory_provider", None)

    # 判断是否为新建会话：如果请求中没有 session_id，则生成一个新的
    session_id = request.session_id or str(uuid.uuid4())
    is_new_session = request.session_id is None

    # ===== Step 1: 会话验证 / 创建 =====
    async with async_session_factory() as db:
        if not is_new_session:
            # 恢复已有会话：验证会话存在且处于 active 状态
            result = await db.execute(
                text("SELECT id, status FROM sessions WHERE id = :id"),
                {"id": session_id},
            )
            row = result.fetchone()
            if row is None:
                # 会话不存在 → 404
                raise HTTPException(status_code=404, detail="Session not found")
            if row[1] == "closed":
                # 会话已关闭 → 400
                raise HTTPException(status_code=400, detail="Session is closed")
        else:
            # 新建会话：从用户消息取前 80 个字符作为会话标题
            title = request.message[:80] + ("..." if len(request.message) > 80 else "")
            await db.execute(
                text("""
                    INSERT INTO sessions (id, user_id, title, status, message_count)
                    VALUES (:id, :user_id, :title, 'active', 0)
                """),
                {"id": session_id, "user_id": request.user_id, "title": title},
            )
            await db.commit()

    # ===== Step 2: 长期记忆召回 =====
    # 从记忆库中检索与当前用户消息相关的历史记忆
    recalled_data = []
    if memory_provider:
        recalled = await memory_provider.recall(
            request.message,
            top_k=MEMORY_RECALL_TOP_K,
            user_id=request.user_id,
        )
        recalled_data = [
            {"content": m.content, "category": m.category}
            for m in recalled
        ]

    # ===== Step 3: 构建初始 Agent 状态 =====
    # 获取聊天模型实例
    model = model_provider.get_chat_model()
    # 将工具绑定到模型上 —— 这启用了 LLM 的 function calling 能力
    # 绑定后，模型在生成回复时就知道有哪些工具可用，以及如何调用它们
    model_with_tools = model.bind_tools(tool_registry.get_llm_tools())

    # 初始状态字典 —— 这是 LangGraph 计算的起点
    initial_state = {
        "messages": [HumanMessage(content=request.message)],  # 用户消息
        "session_id": session_id,         # 当前会话 ID
        "user_id": request.user_id,       # 用户 ID
        "recalled_memories": recalled_data,  # 召回的长期记忆
        "active_skill": None,             # 当前激活的技能（初始为 None）
        "tool_calls": [],                 # 工具调用记录（初始为空）
        "loop_count": 0,                  # 循环计数器
        "final_answer": None,            # 最终答案（Agent 循环结束时填充）
        "compact_summary": None,         # 上下文摘要（用于 token 溢出时压缩）
    }

    # LangGraph 的运行配置
    config = {
        "configurable": {
            "thread_id": session_id,        # 线程 ID = 会话 ID，用于断点续传
            "tool_registry": tool_registry,  # 工具注册表
            "skill_registry": skill_registry, # 技能注册表
            "model": model_with_tools,       # 已绑定工具的模型
        },
    }

    # ===== Step 4: 执行 Agent 循环 =====
    try:
        # 获取或缓存构建 LangGraph 执行图
        # 图（graph）的构建代价较高，首次构建后缓存到 app.state 中
        if not hasattr(app_state, "_graph"):
            from src.agent.graph import build_graph
            app_state._graph = build_graph(
                tool_registry=tool_registry,
                skill_registry=skill_registry,
                checkpoint_saver=app_state.checkpoint_saver,
            )

        # ainvoke 异步执行 Agent 循环：
        # Agent 会经历"思考 → 调用工具 → 观察结果 → 再思考"的
        # 迭代过程，直到产生最终答案或达到最大循环次数
        result = await app_state._graph.ainvoke(initial_state, config)

        # 从执行结果中提取最终答案
        final_answer = result.get("final_answer", "")
        if not final_answer:
            # 如果 final_answer 字段为空，回退到从 messages 列表中
            # 取最后一条 AI 消息的内容
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

    except Exception as e:
        # Agent 循环执行失败时记录错误日志，向用户返回友好提示
        logger.exception("Agent loop execution failed")
        return ChatResponse(
            session_id=session_id,
            error=f"执行失败: {str(e)}",
        )

    # ===== Step 5: 更新会话元数据 =====
    async with async_session_factory() as db:
        await db.execute(
            text("""
                UPDATE sessions
                SET message_count = message_count + 1, updated_at = :now
                WHERE id = :id
            """),
            {"id": session_id, "now": datetime.now(timezone.utc)},
        )
        await db.commit()

    # ===== Step 6: 归档检测 =====
    # 检测用户消息中是否有值得归档为长期记忆的信息
    # 例如用户明确表达偏好、分享关键事实等
    if memory_provider:
        from src.agent.nodes.think import detect_archive_triggers
        triggers = detect_archive_triggers(request.message)
        for t in triggers:
            await memory_provider.archive(
                user_id=request.user_id,
                session_id=session_id,
                content=t["content"],
                category=t["category"],
                source=t["source"],
            )

    return ChatResponse(
        session_id=session_id,
        answer=final_answer or "(无输出)",
        loop_count=result.get("loop_count", 0),
    )

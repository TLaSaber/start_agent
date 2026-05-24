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
    app_state = req.app.state
    tool_registry = app_state.tool_registry
    skill_registry = app_state.skill_registry
    model_provider = app_state.model_provider
    memory_provider = getattr(app_state, "memory_provider", None)

    session_id = request.session_id or str(uuid.uuid4())
    is_new_session = request.session_id is None

    # 1. Validate or create session
    async with async_session_factory() as db:
        if not is_new_session:
            result = await db.execute(
                text("SELECT id, status FROM sessions WHERE id = :id"),
                {"id": session_id},
            )
            row = result.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Session not found")
            if row[1] == "closed":
                raise HTTPException(status_code=400, detail="Session is closed")
        else:
            title = request.message[:80] + ("..." if len(request.message) > 80 else "")
            await db.execute(
                text("""
                    INSERT INTO sessions (id, user_id, title, status, message_count)
                    VALUES (:id, :user_id, :title, 'active', 0)
                """),
                {"id": session_id, "user_id": request.user_id, "title": title},
            )
            await db.commit()

    # 2. Recall long-term memories
    recalled_data = []
    if memory_provider:
        recalled = await memory_provider.recall(
            request.message, top_k=MEMORY_RECALL_TOP_K, user_id=request.user_id
        )
        recalled_data = [
            {"content": m.content, "category": m.category}
            for m in recalled
        ]

    # 3. Build the initial Agent State
    model = model_provider.get_chat_model()
    model_with_tools = model.bind_tools(tool_registry.get_llm_tools())

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "session_id": session_id,
        "user_id": request.user_id,
        "recalled_memories": recalled_data,
        "active_skill": None,
        "tool_calls": [],
        "loop_count": 0,
        "final_answer": None,
        "compact_summary": None,
    }

    config = {
        "configurable": {
            "thread_id": session_id,
            "tool_registry": tool_registry,
            "skill_registry": skill_registry,
            "model": model_with_tools,
        },
    }

    # 4. Execute the Agent Loop
    try:
        # Get or build graph (cached on app state)
        if not hasattr(app_state, "_graph"):
            from src.agent.graph import build_graph
            app_state._graph = build_graph(
                tool_registry=tool_registry,
                skill_registry=skill_registry,
                checkpoint_saver=app_state.checkpoint_saver,
            )

        result = await app_state._graph.ainvoke(initial_state, config)

        # Extract final answer
        final_answer = result.get("final_answer", "")
        if not final_answer:
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
                    break

    except Exception as e:
        logger.exception("Agent loop execution failed")
        return ChatResponse(session_id=session_id, error=f"执行失败: {str(e)}")

    # 5. Update session metadata
    async with async_session_factory() as db:
        await db.execute(
            text("UPDATE sessions SET message_count = message_count + 1, updated_at = :now WHERE id = :id"),
            {"id": session_id, "now": datetime.now(timezone.utc)},
        )
        await db.commit()

    # 6. Archive detection
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

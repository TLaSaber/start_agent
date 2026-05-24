"""
会话（Session）路由 —— 会话的 CRUD 管理与状态流转。

本模块提供了对话会话的增删改查接口。会话（Session）代表一次
连续的对话过程，从创建（active）到关闭（closed）构成了完整的生命周期。

状态流转：
    active（活跃） → closed（已关闭）
    └── 可以继续对话    └── 不可再发送消息
"""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from src.db.engine import async_session_factory

router = APIRouter(prefix="/session", tags=["session"])


async def get_db() -> AsyncSession:
    """FastAPI 依赖注入 —— 提供数据库会话。

    使用 async generator 模式，在请求处理期间提供一个数据库
    会话，请求结束后自动关闭。

    FastAPI 的 Depends 机制：
        在路由函数的参数中声明 db: AsyncSession = Depends(get_db)，
        FastAPI 会自动调用 get_db() 并将结果注入到 db 参数中。
        yield 之前的代码在请求开始时执行，yield 之后的代码在
        请求结束后执行（用于清理资源）。
    """
    async with async_session_factory() as session:
        yield session


@router.get("/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """获取指定会话的详细信息。

    参数：
        session_id: 会话的唯一标识符（UUID 字符串）。

    返回：
        包含会话所有字段的字典：
            - id: 会话 ID
            - user_id: 关联的用户 ID
            - title: 会话标题（由第一条消息自动生成）
            - status: 状态（"active" 或 "closed"）
            - message_count: 已发送的消息数量
            - created_at: 创建时间
            - updated_at: 最后更新时间

    异常：
        404: 如果指定 ID 的会话不存在。
    """
    result = await db.execute(
        text("""
            SELECT id, user_id, title, status, message_count, created_at, updated_at
            FROM sessions WHERE id = :id
        """),
        {"id": session_id},
    )
    row = result.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "id": row[0],
        "user_id": row[1],
        "title": row[2],
        "status": row[3],
        "message_count": row[4],
        "created_at": str(row[5]) if row[5] else None,
        "updated_at": str(row[6]) if row[6] else None,
    }


@router.get("/list")
async def list_sessions(user_id: str, db: AsyncSession = Depends(get_db)):
    """列出指定用户的所有会话。

    注意：这里使用查询参数传递 user_id，而不是路径参数。
    因为 GET 请求路径中不包含 user_id，它通过 ?user_id=xxx
    的形式传递。

    排序规则：按更新时间倒序（最新的在前），最多返回 50 条。

    参数：
        user_id: 查询参数，指定要列出哪个用户的会话。

    返回：
        会话字典列表，每个字典结构与 get_session 相同。
    """
    result = await db.execute(
        text("""
            SELECT id, user_id, title, status, message_count, created_at, updated_at
            FROM sessions WHERE user_id = :uid
            ORDER BY updated_at DESC LIMIT 50
        """),
        {"uid": user_id},
    )
    rows = result.fetchall()
    return [
        {
            "id": row[0],
            "user_id": row[1],
            "title": row[2],
            "status": row[3],
            "message_count": row[4],
            "created_at": str(row[5]) if row[5] else None,
            "updated_at": str(row[6]) if row[6] else None,
        }
        for row in rows
    ]


@router.post("/{session_id}/close")
async def close_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """关闭一个活跃的会话。

    状态流转：
        active（活跃）→ closed（已关闭）

    关闭后的会话不能再接收新的消息（chat 端点会拒绝 closed 状态的会话）。

    实现细节：
        UPDATE 语句中使用了 "status = 'active'" 条件，确保只有
        active 状态的会话会被关闭。如果会话已经 closed 或不存在，
        rowcount 将为 0，返回 404 错误。

    参数：
        session_id: 要关闭的会话 ID。

    返回：
        {"status": "closed"} 表示操作成功。

    异常：
        404: 会话不存在或已经关闭。
    """
    result = await db.execute(
        text("""
            UPDATE sessions
            SET status = 'closed', updated_at = :now
            WHERE id = :id AND status = 'active'
        """),
        {"id": session_id, "now": datetime.now(timezone.utc)},
    )
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(
            status_code=404,
            detail="Session not found or already closed",
        )
    return {"status": "closed"}


@router.delete("/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """删除指定会话（从数据库中永久移除）。

    危险操作：此操作不可逆，会从数据库中完全删除该会话记录。
    不对会话的状态做检查——无论是 active 还是 closed 都会被删除。

    参数：
        session_id: 要删除的会话 ID。

    返回：
        {"deleted": True} 表示操作成功（即使会话不存在也返回 True）。
    """
    await db.execute(
        text("DELETE FROM sessions WHERE id = :id"),
        {"id": session_id},
    )
    await db.commit()
    return {"deleted": True}

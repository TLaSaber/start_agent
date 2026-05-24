from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from src.db.engine import async_session_factory

router = APIRouter(prefix="/session", tags=["session"])


async def get_db() -> AsyncSession:
    async with async_session_factory() as session:
        yield session


@router.get("/{session_id}")
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("SELECT id, user_id, title, status, message_count, created_at, updated_at FROM sessions WHERE id = :id"),
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
    result = await db.execute(
        text("SELECT id, user_id, title, status, message_count, created_at, updated_at FROM sessions WHERE user_id = :uid ORDER BY updated_at DESC LIMIT 50"),
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
    result = await db.execute(
        text("UPDATE sessions SET status = 'closed', updated_at = :now WHERE id = :id AND status = 'active'"),
        {"id": session_id, "now": datetime.now(timezone.utc)},
    )
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Session not found or already closed")
    return {"status": "closed"}


@router.delete("/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)):
    await db.execute(text("DELETE FROM sessions WHERE id = :id"), {"id": session_id})
    await db.commit()
    return {"deleted": True}

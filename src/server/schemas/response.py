from pydantic import BaseModel


class ChatResponse(BaseModel):
    session_id: str
    answer: str | None = None
    error: str | None = None
    loop_count: int = 0


class SessionResponse(BaseModel):
    id: str
    user_id: str
    title: str | None
    status: str
    message_count: int
    created_at: str | None
    updated_at: str | None

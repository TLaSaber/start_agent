from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str | None = None
    user_id: str
    message: str

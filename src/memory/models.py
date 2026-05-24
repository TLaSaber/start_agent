from pydantic import BaseModel


class MemoryEntry(BaseModel):
    id: str
    user_id: str
    session_id: str | None = None
    content: str
    category: str = "fact"  # preference | knowledge | fact
    source: str = "auto_archive"  # auto_archive | user_command | rule_match
    created_at: str | None = None
    ttl_days: int | None = None

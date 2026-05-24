from abc import ABC, abstractmethod
from src.memory.models import MemoryEntry


class MemoryProvider(ABC):
    @abstractmethod
    async def archive(
        self,
        user_id: str,
        content: str,
        category: str = "fact",
        source: str = "auto_archive",
        session_id: str | None = None,
        ttl_days: int | None = None,
    ) -> str:
        """Archive a memory entry, returns memory_id"""
        ...

    @abstractmethod
    async def recall(self, query: str, top_k: int = 5, user_id: str | None = None) -> list[MemoryEntry]:
        """Keyword-based recall of relevant memories"""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        ...

    @abstractmethod
    async def list_by_user(self, user_id: str, limit: int = 50) -> list[MemoryEntry]:
        """List all memories for a user"""
        ...

"""
记忆模块 —— 记忆提供者的抽象接口。

本模块定义了 MemoryProvider 抽象基类，为 PyAgent 提供统一的
记忆存储与检索接口。记忆是 Agent 实现"记住用户偏好和上下文"
能力的关键组件。

设计意图：
    通过抽象接口，将记忆的存储实现（可以是 SQLite、Redis、
    向量数据库等）与业务逻辑解耦。上层代码（Agent 循环）只
    依赖这个抽象接口，不关心底层存储细节。
"""

from abc import ABC, abstractmethod
from src.memory.models import MemoryEntry


class MemoryProvider(ABC):
    """记忆提供者的抽象接口。

    PyAgent 的记忆系统支持两种记忆类型：

    1. 短期记忆（Short-term Memory）
       - 保存在 Agent 的对话上下文中（messages 列表）
       - 随会话存在而存在，会话结束即消失
       - 实现方式：LangGraph 的 checkpoint（检查点）机制

    2. 长期记忆（Long-term Memory）—— 本接口负责的部分
       - 持久化存储在数据库中（如 SQLite）
       - 跨会话存在，Agent 可以在新会话中 recall（召回）历史记忆
       - 用于记住用户的偏好、关键事实、重要知识
       - 实现方式：archive（归档）→ 存储 → recall（召回）

    四个核心方法的职责分工：
        archive     写入新记忆（C - Create）
        recall      基于关键词检索相关记忆（R - Read）
        list_by_user 列出某用户所有记忆（R - Read）
        delete      删除指定记忆（D - Delete）

    这构成了一个完整的记忆 CRUD 接口。
    """

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
        """归档一条新记忆。

        将重要的信息持久化存储，供后续会话中使用。

        参数：
            user_id: 用户标识，记忆归属于该用户。
            content: 记忆内容文本。
            category: 记忆类别，用于分类管理。
                     可取的值： "fact"（事实）、"preference"（偏好）、
                               "knowledge"（知识）。
            source: 记忆来源，标识这条记忆是如何产生的。
                    可取的值： "auto_archive"（自动归档）、
                              "user_command"（用户指令）、
                              "rule_match"（规则匹配）。
            session_id: 可选的会话 ID，追踪记忆来源于哪个会话。
            ttl_days: 可选的过期时间（天），超过此时长后记忆可被清理。
                     为 None 表示永不过期。

        返回：
            新创建的记忆 ID（UUID 字符串），用于后续引用或删除。
        """
        ...

    @abstractmethod
    async def recall(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[MemoryEntry]:
        """基于关键词召回相关记忆。

        这是长期记忆系统的核心方法。当 Agent 收到用户新消息时，
        会调用此方法从已存储的记忆中检索与当前对话相关的信息。

        召回机制说明：
            当前实现使用关键词匹配（LIKE 查询），未来可以升级为
            向量相似度搜索（embedding + cosine similarity）以获得
            更好的语义匹配效果。

        参数：
            query: 查询文本（通常是用户当前的消息），系统会自动
                   从中提取关键词进行匹配。
            top_k: 最多返回的记忆条数，默认 5 条。
            user_id: 可选的用户 ID 过滤。指定后只召回该用户的记忆，
                    不指定则召回所有用户的记忆。

        返回：
            匹配到的 MemoryEntry 列表，按时间倒序排列（最新的在前）。
        """
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """删除一条记忆。

        参数：
            memory_id: 要删除的记忆 ID（archive 方法返回的值）。

        返回：
            True 表示成功删除，False 表示未找到该记忆。
        """
        ...

    @abstractmethod
    async def list_by_user(self, user_id: str, limit: int = 50) -> list[MemoryEntry]:
        """列出指定用户的所有记忆。

        参数：
            user_id: 用户标识。
            limit: 最多返回的记录数，默认 50 条。

        返回：
            该用户的 MemoryEntry 列表，按时间倒序排列。
        """
        ...

"""
记忆模块 —— 数据模型定义。

本模块定义了记忆系统中使用的核心数据类 MemoryEntry，
用于在内存中表示一条记忆记录。
"""

from pydantic import BaseModel


class MemoryEntry(BaseModel):
    """记忆条目数据模型。

    对应数据库中 memories 表的一条记录，也是各方法之间
    传输记忆数据的标准格式。

    各字段含义：
        id: 记忆的唯一标识符（UUID 字符串），由存储层在创建时生成。
        user_id: 归属用户的 ID，用于按用户隔离记忆数据。
        session_id: 可选的会话 ID，标记该记忆产生于哪一次对话会话。
                    可用于追溯上下文。为 None 表示不关联特定会话。
        content: 记忆的文本内容。例如 "用户喜欢用 Python 编写脚本"。

        category: 记忆类别，用于区分不同类型的信息：
            - "fact"（事实）：客观信息，如"用户住在北京"；
            - "preference"（偏好）：用户的主观偏好，如"用户喜欢简洁的回答"；
            - "knowledge"（知识）：用户分享的专业知识或背景信息。
            默认值为 "fact"。

        source: 记忆来源，标识记忆的产生方式：
            - "auto_archive"（自动归档）：系统在对话中自动检测并归档；
            - "user_command"（用户指令）：用户明确要求记住的信息；
            - "rule_match"（规则匹配）：通过预定义规则匹配到的信息。
            默认值为 "auto_archive"。

        created_at: 创建时间的字符串表示。在存储层自动生成，
                    格式通常为 ISO 8601（如 "2024-01-15T10:30:00"）。
                    为 None 表示时间信息不可用。

        ttl_days: 存活时间（天）。超过此天数后，该记忆可以被视为
                  过期并被清理。为 None 表示永不过期。
    """

    id: str
    user_id: str
    session_id: str | None = None
    content: str
    category: str = "fact"
    source: str = "auto_archive"
    created_at: str | None = None
    ttl_days: int | None = None

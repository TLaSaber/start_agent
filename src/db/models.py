"""
数据库 ORM 模型 —— SQLAlchemy 声明式模型定义。

本模块定义了 PyAgent 数据库表结构的 ORM（对象关系映射）模型。
SQLAlchemy 的声明式映射让我们可以用 Python 类的方式来定义
数据库表结构，无需手写 SQL DDL（数据定义语言）。

ORM 的优势：
    1. 使用 Python 代码定义表结构，类型安全；
    2. 自动生成 CREATE TABLE 语句（通过 Base.metadata.create_all）；
    3. 与异步引擎兼容，支持 async/await 语法；
    4. 便于维护和版本控制（表定义就是代码）。
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类。

    所有 ORM 模型类都应继承自此类。
    Base.metadata 收集了所有子类的表定义信息，
    create_all 方法使用这些信息来创建数据库表。

    关于 DeclarativeBase：
        SQLAlchemy 2.0 引入的新式声明基类，替代了旧的
        declarative_base() 函数方式。它使用 Python 原生的
        类继承机制，更加符合 Python 风格。
    """
    pass


class Session(Base):
    """对话会话 ORM 模型 —— 对应数据库中的 sessions 表。

    记录用户与 Agent 之间的一次连续对话会话。每次用户打开
    新对话时都会创建一条会话记录。

    各字段含义：
        id: 会话的唯一标识符（UUID 字符串），作为主键。
            使用 uuid4() 生成随机 UUID，保证全局唯一性。
            类型为 String 而非 UUID 类型，是出于数据库兼容性考虑
            （SQLite 对 UUID 原生支持有限）。

        user_id: 关联的用户 ID，用于按用户隔离会话数据。
                 建立索引（index=True）以加速"查询某用户的所有会话"操作。

        title: 会话标题，由创建会话时用户的第一条消息的前 80 个字符
               自动生成，方便在会话列表中快速识别。可以为 NULL。

        status: 会话状态，仅有两个可能的值：
                - "active"（活跃）：可继续对话
                - "closed"（关闭）：已关闭，不可再发送消息
                默认值为 "active"。

        message_count: 该会话中的消息数量计数。每次用户发送消息后，
                       由 chat 端点自动递增。

        created_at: 会话创建时间（UTC），在插入记录时自动设置。
                    使用 lambda 延迟求值确保每次创建新记录时
                    都获取当前时间。

        updated_at: 最后更新时间（UTC），在插入和更新时自动设置。
                    onupdate 参数确保每次 UPDATE 时自动刷新。
    """

    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=True)
    # 状态字段：'active' | 'closed'
    status = Column(String, nullable=False, default="active")
    message_count = Column(Integer, default=0)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

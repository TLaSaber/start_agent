# Agent 平台 MVP 设计规范

**日期:** 2026-05-24
**状态:** 已确认
**技术栈:** Python 3.12+, LangGraph, FastAPI, SQLAlchemy, DeepSeek (OpenAI 兼容)

---

## 1. 概述

### 1.1 目标

为团队搭建一个通用的 Agent 执行平台。MVP 聚焦核心框架：用户通过 HTTP 请求发送指令，Agent Loop 执行推理与工具调用，返回结果。不限定场景领域，先搭通用骨架，后续按场景迭代。

### 1.2 MVP 范围

| 模块 | 包含 | 不包含 |
|------|------|--------|
| Agent Loop 核心 | Observe → Think → Act 循环 | 专家智能体委派 |
| 会话与上下文 | 会话 CRUD、Checkpoint 持久化、Compact 压缩 | WebSocket 流式推送 |
| 记忆管理 | MemoryProvider 抽象、SQLite 实现、三层归档控制 | 向量语义搜索 |
| 工具与技能 | ToolRegistry、SkillRegistry、8 个内置工具、渐进式加载 | 扩展工具远程执行 |
| 模型防腐 | OpenAI 兼容协议封装、YAML 配置切换 | 多提供商负载均衡 |
| 权限管理 | Act 节点内静态白名单 | 完整风险矩阵 + HITL |
| 韧性管道 | loop_count 兜底、LLM 调用重试 | 声明式执行策略、熔断 |

### 1.3 交互模式

- MVP: HTTP 同步请求-响应（POST /chat），等待完整结果后返回
- 后续: WebSocket 流式推送 + HITL 异步确认

---

## 2. 整体分层架构

```
┌──────────────────────────────────────────────────────┐
│                  🌐 接入层 (FastAPI)                  │
│                                                      │
│  POST /chat          POST /session/close             │
│  GET  /session/{id}  GET  /health                    │
│                                                      │
│  职责：参数校验 · 路由分发 · 响应序列化               │
├──────────────────────────────────────────────────────┤
│                  🧠 Agent Runtime                    │
│                                                      │
│         AgentLoop (LangGraph StateGraph)             │
│                                                      │
│  Observe ──→ Think ──→ Act                           │
│     ▲                      │                         │
│     └────── loop ──────────┘                         │
│                                                      │
│  依赖：SessionStore · MemoryProvider                 │
│        ToolRegistry · SkillRegistry                  │
│                                                      │
│  职责：Agent Loop 生命周期 · 状态编排 · 异常回注     │
├──────────────────────────────────────────────────────┤
│                  🔌 能力层 (Providers)               │
│                                                      │
│  ModelProvider  ·  ToolRegistry  ·  MemoryProvider   │
│  SkillRegistry                                       │
│                                                      │
│  职责：可插拔的底层能力抽象，统一接口，防腐隔离       │
├──────────────────────────────────────────────────────┤
│                  💾 基础设施层                        │
│                                                      │
│  SQLite · 文件系统                                   │
│  (MVP 阶段 SQLite 单文件，后续可切 PG)               │
└──────────────────────────────────────────────────────┘
```

**分层意图:**
- Agent Runtime 可独立测试（无需真实 HTTP 请求）
- 换模型只需改 ModelProvider 实现，不碰 Agent 逻辑
- 后续加 WebSocket 流式，只需改接入层

---

## 3. LangGraph StateGraph 核心设计

### 3.1 AgentState

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]    # LangGraph 原生 reducer
    session_id: str
    user_id: str
    recalled_memories: list[dict]
    active_skill: SkillDefinition | None
    tool_calls: list[ToolCall]
    loop_count: int
    final_answer: str | None
    compact_summary: str | None                # Compact 缓存
```

### 3.2 图拓扑

```
START
  │
  ▼
Observe ──→ Think ──→ Route ──→ Act
  ▲                              │
  └──────────── loop ────────────┘

Route → END (条件: final_answer 非空 或 loop_count 超限)
```

### 3.3 节点职责

**Observe 节点:**
1. 从 checkpoint 加载会话历史 (messages)
2. 检查 token 数，按需触发 Compact 压缩
3. 调用 `memory_provider.recall(query, top_k=3)` 召回长期记忆
4. 注入 System Prompt（含技能清单）
5. 注入召回的记忆作为上下文补充

**Think 节点:**
1. 获取 model_provider.get_chat_model() 实例
2. 绑定 tools schema (由 ToolRegistry 提供)
3. 调用 LLM，解析响应：
   - 若 LLM 返回 `final_answer` → 写入 state
   - 若 LLM 返回 `tool_calls` → 写入 state.tool_calls
4. 归档判断：检查用户消息是否触发归档规则

**Act 节点:**
1. 遍历 state.tool_calls，逐一执行
2. 权限白名单拦截：高风险工具拒绝执行（MVP 阶段 exec_shell 拒绝）
3. 执行结果以 ToolMessage 形式追加到 messages
4. loop_count += 1

**Route (条件边):**
```python
def route_after_think(state: AgentState) -> Literal["act", "__end__"]:
    if state["final_answer"] is not None:
        return "__end__"
    if state["loop_count"] >= MAX_LOOPS:  # 默认 15
        state["final_answer"] = "达到最大循环次数，任务中断"
        return "__end__"
    if state["tool_calls"]:
        return "act"
    return "__end__"
```

---

## 4. 会话与上下文管理

### 4.1 核心决策

Session 持久化复用 LangGraph Checkpoint（SqliteSaver）。每个 session 对应一个 `thread_id`，graph 执行完自动落盘。会话恢复 = checkpoint 重放。

### 4.2 Session 元数据表

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,          -- UUID, 同时用作 LangGraph thread_id
    user_id TEXT NOT NULL,
    title TEXT,                   -- 首条消息截取
    status TEXT NOT NULL DEFAULT 'active',  -- active | closed
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4.3 上下文压缩 (Compact)

在 Observe 节点中触发：
1. 用 tiktoken 估算 context tokens
2. 超过阈值（80% 上下文窗口）时，取最近 6 轮之前的消息生成摘要
3. 摘要作为 SystemMessage 注入，早期消息从上下文移除
4. 摘要缓存到 state.compact_summary，避免重复压缩
5. **不删除** checkpoint 中的全量历史

### 4.4 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /chat | 发送消息 (body: `{session_id?, user_id, message}`) |
| GET | /session/{id} | 查询会话详情 |
| GET | /session/list?user_id= | 用户会话列表 |
| POST | /session/{id}/close | 关闭会话 |
| DELETE | /session/{id} | 删除会话 |

---

## 5. 记忆管理层

### 5.1 MemoryProvider 接口

```python
class MemoryProvider(ABC):
    async def archive(self, entry: MemoryEntry) -> str: ...
    async def recall(self, query: str, top_k: int = 5) -> list[MemoryEntry]: ...
    async def delete(self, memory_id: str) -> bool: ...
    async def list_by_user(self, user_id: str, limit: int = 50) -> list[MemoryEntry]: ...

class MemoryEntry(BaseModel):
    id: str
    user_id: str
    session_id: str | None
    content: str
    category: Literal["preference", "knowledge", "fact"]
    source: Literal["auto_archive", "user_command", "rule_match"]
    created_at: datetime
    ttl_days: int | None
```

### 5.2 短期记忆 vs 长期记忆

- **短期记忆**: 当前会话上下文窗口的消息，Observe 节点自动注入
- **长期记忆**: 跨会话持久化，通过三层控制归档，Observe 阶段召回

### 5.3 三层归档控制

1. 系统默认规则：偏好类永久归档，事实类 30 天 TTL
2. 用户指令："记住这个" → 强制归档
3. 智能体判断：Think 阶段可选标记重要信息归档（需 auto_archive 开关开启）

### 5.4 MVP 实现

SQLite + LIKE/INSTR 关键词匹配。表结构：

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'fact',
    source TEXT NOT NULL DEFAULT 'auto_archive',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ttl_days INTEGER
);
CREATE INDEX idx_memories_user ON memories(user_id);
CREATE INDEX idx_memories_content ON memories(content);
```

后续需要语义搜索时，实现 VectorMemoryProvider，Runtime 代码零改动。

---

## 6. 工具与技能调度层

### 6.1 Tool 基类

```python
class BaseTool(ABC):
    name: str
    description: str
    parameters: dict              # JSON Schema (OpenAI function calling 格式)
    risk_level: Literal["low", "medium", "high", "critical"]

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult: ...

class ToolResult(BaseModel):
    success: bool
    output: str
    error: str | None
```

### 6.2 MVP 内置工具

| 工具 | 功能 | 风险 |
|------|------|------|
| read_file | 读取文件内容 | low |
| write_file | 创建/覆盖文件 | medium |
| list_dir | 列出目录内容 | low |
| search_file | 按名称搜索文件 | low |
| grep_content | 文件内容正则搜索 | low |
| exec_shell | 执行系统命令 | high (MVP 拒绝) |
| http_request | 发起 HTTP 请求 | medium |
| db_query | 执行只读 SQL 查询 | medium |

### 6.3 Skill 定义

```python
class SkillDefinition(BaseModel):
    name: str                    # "code-review"
    summary: str                 # 一句话简介 (清单展示)
    description: str             # 完整提示词 (按需加载)
    tools: list[str]             # 推荐工具白名单
    constraints: list[str]       # 执行约束
    risk_override: dict[str, str] | None
```

### 6.4 渐进式加载

1. Observe: 注入技能清单（名称 + 一句话简介）到 System Prompt
2. Think: LLM 判断是否命中技能 → 输出 skill_activate
3. 下一轮 Observe: 检测到激活，加载完整 Skill 定义注入上下文
4. Act: 检查工具是否在 Skill 白名单内，不在则拒绝

### 6.5 ToolRegistry 自动发现

```python
class ToolRegistry:
    def discover(self, tool_dir: str = "agent/tools") -> None:
        """扫描目录，自动注册所有 BaseTool 子类"""

    def get_llm_tools(self) -> list[dict]:
        """生成 OpenAI function calling 格式的 tools 列表"""

    async def execute(self, name: str, **kwargs) -> ToolResult: ...
```

---

## 7. 模型防腐层

### 7.1 ModelProvider 接口

```python
class ModelProvider(ABC):
    def get_chat_model(self, model_name: str | None = None) -> BaseChatModel: ...
    def get_available_models(self) -> list[ModelInfo]: ...
    def count_tokens(self, text: str, model: str | None = None) -> int: ...
```

### 7.2 OpenAICompatProvider

基于 LangChain `ChatOpenAI` 的薄封装。通过 `api_base` 和 `api_key` 配置对接任何 OpenAI 兼容服务（DeepSeek、OpenAI、Ollama 等）。

### 7.3 配置驱动

```yaml
# config/model.yaml
providers:
  deepseek:
    type: openai_compat
    api_base: "${DEEPSEEK_API_BASE}"
    api_key: "${DEEPSEEK_API_KEY}"
    default_model: "deepseek-v3"
    available_models:
      - name: "deepseek-v3"
        max_tokens: 65536
        capabilities: [chat, function_calling]

routing:
  main_agent:
    provider: deepseek
    model: deepseek-v3
```

换模型 = 改 YAML + 设环境变量，代码零改动。

---

## 8. 项目目录结构

```
pyagent/
├── pyproject.toml
├── config/
│   ├── model.yaml
│   ├── skills.yaml
│   └── settings.py
├── src/
│   ├── server/           # 接入层: FastAPI
│   │   ├── app.py
│   │   ├── routers/
│   │   │   ├── chat.py
│   │   │   └── session.py
│   │   └── schemas/
│   │       ├── request.py
│   │       └── response.py
│   ├── agent/            # Agent Runtime: LangGraph
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── nodes/
│   │   │   ├── observe.py
│   │   │   ├── think.py
│   │   │   └── act.py
│   │   └── compact.py
│   ├── tools/            # 能力层: 工具
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── builtin/
│   │   │   ├── file_ops.py
│   │   │   ├── search.py
│   │   │   ├── shell.py
│   │   │   ├── http.py
│   │   │   └── database.py
│   │   └── extension/
│   ├── skills/           # 能力层: 技能
│   │   ├── registry.py
│   │   └── definitions/
│   ├── memory/           # 能力层: 记忆
│   │   ├── provider.py
│   │   ├── sqlite_provider.py
│   │   └── models.py
│   ├── models/           # 能力层: 模型防腐
│   │   ├── provider.py
│   │   └── openai_compat.py
│   └── db/               # 基础设施层
│       ├── engine.py
│       └── models.py
└── tests/
    ├── test_graph.py
    ├── test_nodes/
    ├── test_tools/
    ├── test_memory/
    └── test_api/
```

### 模块边界规则

| 模块 | 可依赖 | 不可依赖 |
|------|--------|----------|
| server/ | agent, tools, models, memory, db | — |
| agent/ | tools, models, memory, db | server |
| tools/ | db (仅 database 工具) | server, agent |
| memory/ | db | server, agent |
| models/ | — (纯配置驱动) | server, agent, db |

---

## 9. 错误处理

### 9.1 LLM 调用错误

- 网络错误 → 重试 3 次，指数退避（1s, 2s, 4s）
- API 错误（4xx）→ 直接返回错误信息给用户
- 超时（60s）→ 返回 "模型响应超时"

### 9.2 工具执行错误

- 工具抛出异常 → 捕获，封装为 ToolResult(success=False)，回注到 messages
- 工具超时（30s）→ 强制中断，返回超时错误

### 9.3 循环保护

- loop_count 默认上限 15，超限后强制终止并返回摘要
- 降级策略：若 3 次循环内 LLM 持续调用同一工具且结果相同，终止循环

### 9.4 会话错误

- session_id 不存在 → 返回 404
- 已关闭的 session → 返回 400 "session closed"
- checkpoint 损坏 → 返回 500，记录日志

---

## 10. 技术依赖

```toml
[project]
name = "pyagent"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "langgraph>=0.3",
    "langchain>=0.3",
    "langchain-openai>=0.3",
    "sqlalchemy>=2.0",
    "aiosqlite>=0.20",
    "tiktoken>=0.8",
    "pyyaml>=6.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "httpx>=0.28",
]
```

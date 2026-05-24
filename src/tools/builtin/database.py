"""
数据库查询工具模块。

本模块提供了数据库只读查询的能力，让 LLM 可以查询数据库中的信息。

为什么只允许 SELECT/PRAGMA？
--------------------------
这是一个重要的安全设计决策。
数据库工具如果不加限制，将非常危险：
1. 允许 DELETE/UPDATE/INSERT 意味着可以修改或删除数据
2. 允许 DROP TABLE 意味着可以删除整个表
3. 允许任意 SQL 意味着可以被 SQL 注入攻击利用

因此，DbQueryTool 在 SQL 执行前会进行严格的检查：
- 只允许以 SELECT 或 PRAGMA 开头的语句
- 其他所有语句（INSERT、UPDATE、DELETE、DROP、ALTER、CREATE 等）都会被拒绝
- PRAGMA 也被允许，因为它是 SQLite 中查询数据库信息的只读指令

关于 SQLite 的说明：
------------------
当前实现使用 Python 内置的 sqlite3 模块，只支持 SQLite 数据库。
sqlite3 是 Python 标准库的一部分，不需要额外安装数据库服务。
连接字符串格式：sqlite:///path/to/database.db
"""

from src.tools.base import BaseTool, ToolResult


class DbQueryTool(BaseTool):
    """
    数据库只读查询工具。

    【功能】
    连接到 SQLite 数据库并执行只读的 SELECT 或 PRAGMA 查询，返回格式化的查询结果。
    支持 sqlite:/// 格式的连接字符串。

    【使用场景】
    - 查询数据库中的记录
    - 检查表结构（PRAGMA table_info）
    - 统计数据行数
    - 调试和数据分析

    【安全等级】
    risk_level = "medium"：虽然限制了只读操作，但仍然：
    - 可以读取数据库中的所有数据（包括敏感信息）
    - 大数据量查询可能影响数据库性能
    - 因此需要一定程度的管控

    【parameters 说明】
    - db_url: string, 必需。数据库连接字符串。
             格式: sqlite:///数据库文件路径
             例如: sqlite:///C:/data/mydb.db
    - query: string, 必需。SQL 查询语句（只允许 SELECT 和 PRAGMA）。

    当前限制：
    - 仅支持 SQLite 数据库
    - 仅允许 SELECT 和 PRAGMA 语句
    - 最多返回 100 行数据
    """
    name = "db_query"
    description = "执行只读 SQL 查询。参数 db_url: 数据库连接串, query: SELECT 语句。"
    parameters = {
        "type": "object",
        "properties": {
            "db_url": {"type": "string", "description": "数据库连接串，如 sqlite:///path/to/db"},
            "query": {"type": "string", "description": "只读 SELECT 查询语句"},
        },
        "required": ["db_url", "query"],
    }
    risk_level = "medium"

    async def execute(self, db_url: str = "", query: str = "", **kwargs) -> ToolResult:
        """
        执行数据库查询。

        实现流程：
        1. SQL 安全检查 —— 验证是否只包含 SELECT/PRAGMA 语句
        2. 解析连接字符串 —— 从 sqlite:/// 格式中提取文件路径
        3. 连接数据库 —— 使用 sqlite3.connect()
        4. 执行查询 —— 获取列名和所有结果行
        5. 格式化输出 —— 对齐列名和数值，限制最大行数

        格式化的实现细节：
        -----------------
        - col_names = [desc[0] for desc in cursor.description]
          cursor.description 是查询结果的列信息，每个元素是一个元组，
          第一个元素（索引0）就是列名。
        - " | ".join(col_names): 用 " | " 分隔列名
        - "-" * len(...): 列名下面的分隔线，长度和列名行一致
        - 每行数据也用 " | " 分隔，形成表格效果

        输出示例：
            id | name | age
            -----------------
            1 | Alice | 25
            2 | Bob | 30
            ... (仅显示前 100 行，共 200 行)

        安全机制：
        --------
        query.strip().upper() 将查询转为大写后检查前缀：
        - 不区分大小写（SELECT = select = Select）
        - 只检查前缀，所以在 SELECT 后面可以有任何内容
        - PRAGMA 是 SQLite 特有的只读指令，用于获取元数据

        Args:
            db_url: 数据库连接字符串（如 sqlite:///path/to/db）
            query: SQL 查询语句

        Returns:
            ToolResult: 格式化的查询结果表格，最多 100 行。
                        如果查询无返回行，提示 "(查询成功，无返回行)"。
        """
        # 安全检查：将 SQL 转换为大写并去除首尾空格
        query_stripped = query.strip().upper()
        if not query_stripped.startswith("SELECT") and not query_stripped.startswith("PRAGMA"):
            return ToolResult(
                success=False, output="",
                error=f"仅允许只读查询(SELECT/PRAGMA)，收到: {query[:50]}"
            )

        try:
            import sqlite3
            # 解析连接字符串：从 sqlite:///path/to/db 中提取 /path/to/db
            if db_url.startswith("sqlite:///"):
                db_path = db_url[len("sqlite:///"):]
            elif "///" in db_url:
                db_path = db_url.split("///", 1)[1]
            else:
                db_path = db_url

            # 连接数据库并设置 row_factory 为 sqlite3.Row
            # sqlite3.Row 允许通过列名访问值（像字典一样），而不仅限于索引
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return ToolResult(success=True, output="(查询成功，无返回行)")

            # 获取列名列表
            col_names = [desc[0] for desc in cursor.description]
            # 生成表头行（列名用 | 分隔）
            lines = [" | ".join(col_names), "-" * len(" | ".join(col_names))]
            # 最多显示 100 行
            for row in rows[:100]:
                lines.append(" | ".join(str(v) for v in row))

            output = "\n".join(lines)
            # 如果数据超过 100 行，提示用户还有更多数据
            if len(rows) > 100:
                output += f"\n\n... (仅显示前 100 行，共 {len(rows)} 行)"
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

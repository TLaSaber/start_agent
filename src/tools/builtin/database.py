from src.tools.base import BaseTool, ToolResult


class DbQueryTool(BaseTool):
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
        query_stripped = query.strip().upper()
        if not query_stripped.startswith("SELECT") and not query_stripped.startswith("PRAGMA"):
            return ToolResult(
                success=False, output="",
                error=f"仅允许只读查询(SELECT/PRAGMA)，收到: {query[:50]}"
            )

        try:
            import sqlite3
            if db_url.startswith("sqlite:///"):
                db_path = db_url[len("sqlite:///"):]
            elif "///" in db_url:
                db_path = db_url.split("///", 1)[1]
            else:
                db_path = db_url

            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return ToolResult(success=True, output="(查询成功，无返回行)")

            col_names = [desc[0] for desc in cursor.description]
            lines = [" | ".join(col_names), "-" * len(" | ".join(col_names))]
            for row in rows[:100]:
                lines.append(" | ".join(str(v) for v in row))

            output = "\n".join(lines)
            if len(rows) > 100:
                output += f"\n\n... (仅显示前 100 行，共 {len(rows)} 行)"
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

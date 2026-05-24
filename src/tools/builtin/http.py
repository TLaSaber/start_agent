import urllib.request
import urllib.error
import json
from src.tools.base import BaseTool, ToolResult


class HttpRequestTool(BaseTool):
    name = "http_request"
    description = "发起 HTTP 请求。参数 url: 请求URL, method: GET/POST, headers: JSON字符串(可选), body: 请求体(可选)。"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "请求 URL"},
            "method": {"type": "string", "description": "HTTP 方法: GET 或 POST", "default": "GET"},
            "headers": {"type": "string", "description": "JSON 格式的请求头 (可选)"},
            "body": {"type": "string", "description": "请求体 (可选)"},
        },
        "required": ["url", "method"],
    }
    risk_level = "medium"

    async def execute(self, url: str = "", method: str = "GET", headers: str = "{}", body: str = "", **kwargs) -> ToolResult:
        try:
            parsed_headers = json.loads(headers) if headers else {}
            data = body.encode("utf-8") if body else None

            req = urllib.request.Request(url, data=data, headers=parsed_headers, method=method.upper())
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_body = resp.read().decode("utf-8", errors="replace")
                return ToolResult(
                    success=True,
                    output=f"Status: {resp.status}\n\n{response_body[:5000]}"
                )
        except urllib.error.HTTPError as e:
            return ToolResult(success=False, output="", error=f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            return ToolResult(success=False, output="", error=f"URL 错误: {e.reason}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

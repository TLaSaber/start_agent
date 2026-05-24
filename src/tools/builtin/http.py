"""
HTTP 请求工具模块。

本模块提供了发起 HTTP 请求的能力，让 LLM 可以与网络上的 API 和服务交互。

urllib 库说明（面向新手）：
--------------------------
Python 标准库中自带的 HTTP 客户端库，无需额外安装。
本工具使用 urllib 而非流行的 requests 库，原因是：
1. urllib 是 Python 内置的 —— 零依赖，安装 PyAgent 后直接可用
2. 对于简单的 GET/POST 请求，urllib 完全够用
3. 减少了项目的第三方依赖

urllib 的基本使用流程：
    1. 创建 Request 对象：封装 URL、请求头、请求体、请求方法
    2. 发送请求：urlopen(req) 返回响应对象
    3. 读取响应：resp.read() 获取响应体字节
    4. 解码：decode("utf-8") 将字节转为字符串

安全注意事项：
------------
- 只能发起 HTTP/HTTPS 请求，不能操作文件系统
- 建议限制访问内网地址（如 127.0.0.1、10.x.x.x 等），防止 SSRF 攻击
- 响应体截断为 5000 字符，防止响应太大消耗过多 token
"""

import urllib.request
import urllib.error
import json
from src.tools.base import BaseTool, ToolResult


class HttpRequestTool(BaseTool):
    """
    HTTP 请求工具。

    【功能】
    发起 HTTP 请求到指定的 URL，支持 GET 和 POST 方法。
    可以自定义请求头和请求体。

    【使用场景】
    - 调用外部 REST API 获取数据
    - 向 Web 服务发送数据
    - 检测网站是否可访问
    - 与 AI 服务交互（如调用其他 AI 模型的 API）

    【安全等级】
    risk_level = "medium"：可以访问网络，可能的风险包括：
    - 访问内部服务（SSRF 攻击）
    - 发送大量数据（消耗带宽）
    - 调用付费 API（产生费用）
    - 暴露敏感信息（在 URL 或请求头中）

    【parameters 说明】
    - url: string, 必需。请求的目标 URL（如 https://api.example.com/data）。
    - method: string, 必需。HTTP 方法，目前支持 "GET" 和 "POST"。
    - headers: string, 可选。JSON 字符串格式的请求头（如 '{"Authorization": "Bearer xxx"}'）。
    - body: string, 可选。请求体内容（POST 请求时使用）。
    """
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
        """
        执行 HTTP 请求。

        实现流程：
        1. headers 参数是 JSON 字符串，先用 json.loads 解析为字典
        2. 如果 body 不为空，编码为 UTF-8 字节
        3. 构造 urllib.request.Request 对象
        4. 发送请求并获取响应（30 秒超时）
        5. 读取响应体并解码为字符串（截取前 5000 字符）
        6. 返回 HTTP 状态码和响应内容

        urllib.request.Request 的构造函数：
        ---------------------------------
        Request(url, data=data, headers=headers, method=method)
        - url: 请求地址
        - data: POST 请求的请求体（bytes 类型，GET 请求为 None）
        - headers: 请求头字典
        - method: 请求方法字符串（如 "GET"、"POST"）

        urllib.request.urlopen 的用法：
        ------------------------------
        urlopen(req, timeout=30)
        - req: Request 对象
        - timeout: 超时秒数
        - 返回值是 HTTPResponse 对象，可用作上下文管理器（with 语句）
        - resp.status: HTTP 状态码（200=成功，404=未找到，500=服务器错误等）
        - resp.read(): 读取响应体字节

        Args:
            url: 请求 URL
            method: HTTP 方法（GET 或 POST）
            headers: JSON 格式的请求头字符串
            body: 请求体字符串

        Returns:
            ToolResult:
                - success=True 时 output 包含 "Status: 状态码" 和响应内容
                - success=False 时 error 包含详细的错误描述

        异常处理：
        - HTTPError: HTTP 错误响应（如 404、500），包含状态码和原因
        - URLError: URL 错误（如 DNS 解析失败、连接拒绝）
        - Exception: 其他意外错误

        注意：响应内容截取前 5000 字符，防止输出太长
        """
        try:
            # 解析 headers（JSON 字符串 -> Python 字典）
            parsed_headers = json.loads(headers) if headers else {}
            # 编码请求体（字符串 -> UTF-8 字节）
            data = body.encode("utf-8") if body else None

            req = urllib.request.Request(url, data=data, headers=parsed_headers, method=method.upper())
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_body = resp.read().decode("utf-8", errors="replace")
                return ToolResult(
                    success=True,
                    output=f"Status: {resp.status}\n\n{response_body[:5000]}"
                )
        except urllib.error.HTTPError as e:
            # HTTP 错误响应（如 404 Not Found, 500 Internal Server Error）
            return ToolResult(success=False, output="", error=f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            # URL 错误（如 DNS 解析失败、连接被拒绝）
            return ToolResult(success=False, output="", error=f"URL 错误: {e.reason}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

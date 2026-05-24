"""
工具注册中心模块。

本模块实现了"注册中心（Registry）"设计模式，这是 PyAgent 工具系统的核心管理组件。

为什么需要注册中心？
-----------------
在一个 AI Agent 系统中，可能会注册几十个甚至上百个工具。如果没有一个统一的管理
机构，就会面临以下问题：
  1. 工具散落在各处，难以统一管理
  2. LLM 需要一份所有工具的清单才知道"我能调用什么"
  3. 调用工具时需要一个统一的入口，而非直接实例化工具类

ToolRegistry 的职责：
  1. 注册（register）：接收一个工具实例，将其纳入管理
  2. 查找（get/list_all）：按名称查找或列出所有已注册的工具
  3. 转换（get_llm_tools）：将内部工具列表转换为 LLM 可理解的格式
  4. 执行（execute）：根据工具名称自动找到对应的工具并执行

工作流程：
  1. 应用启动时，创建 ToolRegistry 实例
  2. 将所有工具（ReadFileTool、WriteFileTool 等）注册到 registry 中
  3. 当需要和 LLM 对话时，调用 get_llm_tools() 获取工具列表
  4. LLM 决定调用哪个工具，返回工具名称和参数
  5. 调用 registry.execute(name, **kwargs) 执行工具
  6. 工具执行结果返回给 LLM 进行下一步处理
"""

from src.tools.base import BaseTool, ToolResult


class ToolNotFoundError(Exception):
    """
    工具未找到异常。

    当调用 execute() 时传入了一个未注册的工具名称，就会抛出此异常。
    这个异常继承了 Python 内置的 Exception 类，所以可以被 try/except 捕获。

    典型场景：
    - LLM 返回了一个不存在的工具名（可能是幻觉）
    - 拼写错误，如 "read-flie" 而不是 "read_file"
    - 工具尚未注册就被调用

    使用示例：
        try:
            result = await registry.execute("read_file", path="test.txt")
        except ToolNotFoundError as e:
            print(f"工具不存在: {e}")
    """
    pass


class ToolRegistry:
    """
    工具注册中心 —— 所有工具的"大管家"。

    这个类维护了一个工具名称到工具实例的字典映射（_tools），
    并提供了一系列方法来管理和使用这些工具。

    可以把它想象成一个"电话簿"：
    - register() 就像在电话簿里添加一个联系人
    - get() 就像根据姓名查找电话号码
    - execute() 就像拨打电话 —— 你只要说出名字，它自动帮你接通
    - get_llm_tools() 就像把电话簿整理成一份"给对方看的格式"

    为什么 register() 接收工具实例而不是工具类？
    因为同一个类可以有不同的配置（比如不同的参数），所以我们需要具体的实例。
    """

    def __init__(self):
        """
        初始化空的注册中心。

        内部维护一个字典 _tools，键是工具名称（str），值是工具实例（BaseTool）。
        使用字典的原因是查找效率高（O(1)时间复杂度）。

        Python 小知识：
        - 属性名前的下划线 _tools 表示"内部使用，请勿直接访问"
        - 这只是一种约定（称为"名称修饰"），Python 并不会真的禁止外部访问
        - 但好的编程习惯是：尊重这种约定，通过 register/get/execute 方法来操作
        """
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        注册一个工具实例。

        将工具添加到注册中心的字典中，以 tool.name 为键。
        如果重复注册相同名称的工具，后面的会覆盖前面的（字典的特性）。

        Args:
            tool: BaseTool 的实例（如 ReadFileTool()、WriteFileTool() 等）。
                  注意传入的是实例而不是类，所以要加括号：register(ReadFileTool())

        示例:
            registry = ToolRegistry()
            registry.register(ReadFileTool())    # 注册读文件工具
            registry.register(WriteFileTool())   # 注册写文件工具

        注意：
        - 注册时不会校验工具是否能正常工作
        - 如果两个工具同名，后注册的会覆盖先注册的，使用时需要注意
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """
        根据名称获取已注册的工具实例。

        Args:
            name: 工具名称（即 BaseTool.name 字段的值）

        Returns:
            BaseTool | None: 如果找到则返回工具实例，否则返回 None。
                            返回 None 表示该名称的工具未注册。

        使用字典的 .get() 方法而不是直接索引 []，是因为如果键不存在，
        .get() 会返回 None 而不会抛出 KeyError 异常。
        """
        return self._tools.get(name)

    def list_all(self) -> list[BaseTool]:
        """
        获取所有已注册的工具列表。

        Returns:
            list[BaseTool]: 包含所有已注册工具实例的列表。
                            每次调用都返回一个新列表（因为使用了 list() 转换），
                            所以外部修改返回的列表不会影响注册中心内部的数据。

        用途：
        - 调试时查看当前注册了哪些工具
        - 在管理界面中展示所有可用工具
        """
        return list(self._tools.values())

    def get_llm_tools(self) -> list[dict]:
        """
        将已注册的工具列表转换为 OpenAI Function Calling 格式。

        这是本工具系统中最关键的转换方法之一。OpenAI（以及兼容的 LLM API）
        使用特定的 JSON 格式来描述"可调用的函数"（也就是我们的工具）。

        转换格式说明：
        -------------
        OpenAI Function Calling 要求每个工具描述为一个包含 type 和 function
        字段的字典。具体结构如下：

            {
                "type": "function",           # 固定值，告诉 OpenAI 这是函数调用
                "function": {
                    "name": "工具名称",         # LLM 通过名字来调用
                    "description": "工具描述",   # LLM 根据描述决定用哪个
                    "parameters": { ... }       # JSON Schema，LLM 据此生成参数
                }
            }

        为什么需要这个转换？
        因为 LLM 不理解 Python 对象，只理解 JSON！
        我们的 BaseTool 是 Python 类，必须序列化成 JSON 格式 LLM 才能"看懂"。

        Returns:
            list[dict]: 符合 OpenAI Function Calling 格式的工具描述列表。
                        这个列表会作为参数传给 LLM API，告诉 LLM 它可以调用哪些工具。

        示例输出:
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "读取指定路径的文件内容",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "文件路径"}
                            },
                            "required": ["path"]
                        }
                    }
                },
                ...
            ]
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    async def execute(self, name: str, **kwargs) -> ToolResult:
        """
        异步执行指定名称的工具。

        这是工具调用的统一入口。调用者只需要提供工具名称和参数，
        不需要关心工具的具体实现细节。

        执行流程：
        1. 根据 name 在 _tools 字典中查找工具
        2. 如果未找到，抛出 ToolNotFoundError
        3. 如果找到，调用工具的 execute() 方法并返回结果

        Args:
            name: 要执行的工具名称（必须已经注册）。
            **kwargs: 传递给工具 execute() 方法的关键字参数。
                      具体需要哪些参数，由对应工具的 parameters 字段定义。

        Returns:
            ToolResult: 工具执行结果（包含 success/output/error 三个字段）。

        Raises:
            ToolNotFoundError: 如果 name 对应的工具未注册。

        使用示例:
            result = await registry.execute("read_file", path="/tmp/test.txt")
            if result.success:
                print(result.output)  # 输出文件内容
            else:
                print(f"错误: {result.error}")  # 输出错误信息

        注意：
        - 这个方法是 async 的，调用时必须使用 await
        - 方法本身会抛出异常（ToolNotFoundError），但工具执行过程中的异常
          会被工具内部捕获并包装成 ToolResult 返回
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Tool '{name}' not found")
        return await tool.execute(**kwargs)

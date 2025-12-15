from mcp.server.fastmcp import FastMCP
from context_schema import MCPContextSchema

class ExtendedFastMCP(FastMCP):
    def __init__(self, name: str):
        super().__init__(name)
        self._local_tools = {}
        self.context = MCPContextSchema()

    def tool(self, name: str | None = None, **options):
        """
        扩展 tool 装饰器：
        - 正常注册到 MCP（客户端可见）
        - 同时保存到 _local_tools，方便 Agent 内部调用
        """
        def decorator(func):
            tool_name = name or func.__name__

            # 调用 FastMCP 的原始注册逻辑
            super_decorator = super(ExtendedFastMCP, self).tool(name=name, **options)
            wrapped = super_decorator(func)

            # 本地可调用注册
            self._local_tools[tool_name] = wrapped

            return wrapped
        return decorator

    async def call_tool(self, name: str, **kwargs):
        """
        本地调用 MCP 工具
        """
        if name in self._local_tools:
            func = self._local_tools[name]
            result = func(**kwargs)
            if hasattr(result, "__await__"):  # async 函数
                return await result
            return result
        raise RuntimeError(f"工具未注册: {name}")


# 创建全局 MCP 服务实例
mcp_server = ExtendedFastMCP("GIS-MCP-Server")

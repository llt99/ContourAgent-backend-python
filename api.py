# # import base64
# # import traceback
# # from fastapi import FastAPI
# # from decimal import Decimal
# # import numpy as np
# # from fastapi.responses import JSONResponse
# # from pydantic import BaseModel
# # from nlp_processor import parse_task, to_agent_plan
# # from data_query import DataRetrievalAgent
# # from kriging import InterpolatorAgent
# # from image import MapRenderAgent, ImageAgent
# # from feedback_processor import FeedbackAgent as FeedbackParser
# #
# # # ------------------------
# # # Agent 抽象类
# # # ------------------------
# # class Agent:
# #     def run(self, context: dict) -> dict:
# #         raise NotImplementedError("Agent 必须实现 run 方法")
# #
# #
# # # ------------------------
# # # Agent 实现
# # # ------------------------
# # class NLPAgent(Agent):
# #     def run(self, context: dict) -> dict:
# #         task = parse_task(context["text"])
# #         plan = to_agent_plan(task)
# #         context["task"] = task
# #         context["plan"] = plan
# #
# #         print("🔹 NLP解析结果:", task.model_dump() if hasattr(task, "model_dump") else task.__dict__)
# #
# #         return context
# #
# #
# # class DataAgent(Agent):
# #     def run(self, context: dict) -> dict:
# #         step = context["plan"]["pipeline"][0]
# #         params = step.get("params", {})
# #
# #         if context.get("data_points") and context.get("feedback"):
# #             print(f"🔹 复用历史数据 {len(context['data_points'])} 条")
# #             return context
# #
# #         task_dict = {
# #             "region": params.get("region"),
# #             "formation": params.get("formation"),
# #             "variable": params.get("variable")
# #         }
# #
# #         agent = DataRetrievalAgent()
# #         data_points = agent.execute(task_dict)
# #
# #         context["data_points"] = data_points
# #         print(f"🔹 数据检索结果 {len(data_points)} 条")
# #         return context
# #
# #
# # class KrigingAgent(Agent):
# #     FIELD_MAP = {
# #         "地层厚度": "thickness",
# #         "岩性": "lithology",
# #         # 可以继续添加其他 NLP 变量 -> 数据字段映射
# #     }
# #
# #     def __init__(self):
# #         self.interpolator_agent = InterpolatorAgent()
# #
# #     @staticmethod
# #     def get_coord(point: dict, keys: list):
# #         """从 point 中取第一个存在的字段值"""
# #         for k in keys:
# #             if k in point:
# #                 return point[k]
# #         raise KeyError(f"None of {keys} found in {point}")
# #
# #     def run(self, context: dict) -> dict:
# #         step = context["plan"]["pipeline"][1]
# #         k_params = step.get("params", {})
# #
# #         data_points = context["data_points"]
# #         task = context["task"]
# #
# #         # 映射 NLP 解析字段到数据字段
# #         variable_field = self.FIELD_MAP.get(task.variable, task.variable)
# #
# #         # 构造插值任务字典，兼容多字段名
# #         task_dict = {
# #             "method": k_params.get("method", "ok"),
# #             "variogram_model": k_params.get("model", "spherical"),
# #             "autoOptimizeModel": k_params.get("auto_optimize", False),
# #             "smoothSigma": k_params.get("sigma", 0),
# #             "drift": k_params.get("drift", "linear"),
# #             "points": [
# #                 {
# #                     "lon": self.interpolator_agent.get_coord(p, ["lon", "geo_X", "longitude"]),
# #                     "lat": self.interpolator_agent.get_coord(p, ["lat", "geo_Y", "latitude"]),
# #                     "value": self.interpolator_agent.get_coord(p, [variable_field, "value"])
# #                 }
# #                 for p in data_points
# #             ]
# #         }
# #
# #         kriging_result = self.interpolator_agent.handle(task_dict)
# #
# #         if "error" in kriging_result:
# #             raise Exception(kriging_result["error"])
# #
# #         context["kriging_result"] = kriging_result
# #         print("🔹 插值计算完成")
# #         return context
# #
# #
# # class FeedbackAgent(Agent):
# #     def __init__(self):
# #         self.agent = FeedbackParser()  # 实例化 Agent-4
# #
# #     def run(self, context: dict) -> dict:
# #         feedback_text = context.get("feedback")
# #         if not feedback_text:
# #             return context
# #
# #         # 调用 Agent-4 的 run 方法，得到 MCPContext
# #         mcp_context = self.agent.run(feedback_text)
# #         feedback = mcp_context.params  # 获取结构化字典
# #         print("🔹 用户反馈解析:", feedback)
# #
# #         # 修改 MCP plan
# #         plan = context.get("plan", {})
# #         if "model" in feedback and len(plan.get("pipeline", [])) > 1:
# #             plan["pipeline"][1]["params"]["model"] = feedback["model"]
# #         if "region" in feedback and len(plan.get("pipeline", [])) > 0:
# #             plan["pipeline"][0]["params"]["region"] = feedback["region"]
# #         if feedback.get("re_execute"):
# #             context["re_execute"] = True
# #         if "colormap" in feedback:
# #             context["colormap"] = feedback["colormap"]
# #
# #         # 保存修改后的 plan 和解析结果
# #         context["plan"] = plan
# #         context["feedbackParsed"] = feedback
# #         return context
# #
# #
# # # ------------------------
# # # MCP Controller
# # # ------------------------
# # class MCPController:
# #     def __init__(self):
# #         self.last_context = {}
# #
# #         self.agents = {
# #             "nlp": NLPAgent(),
# #             "feedback": FeedbackAgent(),
# #             "data": DataAgent(),
# #             "kriging": KrigingAgent(),
# #             "image": ImageAgent(),
# #         }
# #
# #     def run(self, context: dict) -> dict:
# #         # 每次都更新 task 和 plan，不复用上一次 NLP 结果
# #         context = self.agents["nlp"].run(context)
# #         # 反馈处理
# #         context = self.agents["feedback"].run(context)
# #         # 数据检索
# #         context = self.agents["data"].run(context)
# #         if not context.get("data_points"):
# #             return context
# #         # 插值
# #         context = self.agents["kriging"].run(context)
# #         # 图件生成
# #         context = self.agents["image"].run(context)
# #         # 缓存最新上下文
# #         self.last_context = context
# #         return context
# #
# #
# # # ------------------------
# # # FastAPI
# # # ------------------------
# # app = FastAPI()
# # mcp = MCPController()
# #
# #
# # class TaskRequest(BaseModel):
# #     text: str
# #     feedback: str | None = None
# #
# #
# # def convert_to_json_serializable(obj):
# #     """递归把 dict/list 中的 Decimal 和 ndarray 转为 float / list"""
# #     if isinstance(obj, list):
# #         return [convert_to_json_serializable(x) for x in obj]
# #     elif isinstance(obj, dict):
# #         return {k: convert_to_json_serializable(v) for k, v in obj.items()}
# #     elif isinstance(obj, Decimal):
# #         return float(obj)
# #     elif isinstance(obj, np.ndarray):
# #         return obj.tolist()
# #     else:
# #         return obj
# #
# # @app.post("/task")
# # async def run_task(req: TaskRequest):
# #     try:
# #         context = {
# #             "text": req.text,
# #             "feedback": req.feedback,
# #             "task": mcp.last_context.get("task"),
# #             "plan": mcp.last_context.get("plan"),
# #             "data_points": mcp.last_context.get("data_points"),
# #         }
# #
# #         context = mcp.run(context)
# #
# #         print(context["image_result"][:50])
# #
# #         return JSONResponse({
# #             "nlpResult": context["task"].model_dump() if context.get("task") else None,
# #             "plan": context.get("plan"),
# #             "dataResult": convert_to_json_serializable(context.get("data_points")),
# #             "krigingResult": convert_to_json_serializable(context.get("kriging_result")),
# #             "imageResult": context.get("image_result"),
# #             "feedbackUsed": req.feedback,
# #             "feedbackParsed": context.get("feedbackParsed")
# #         })
# #
# #     except Exception as e:
# #         print("❌ MCP 执行异常:", str(e))
# #         traceback.print_exc()
# #         return JSONResponse(
# #             status_code=500,
# #             content={"error": str(e), "trace": traceback.format_exc()}
# #         )
# #
# #
# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
#
#
#
# import traceback
# from fastapi import FastAPI
# from decimal import Decimal
# import numpy as np
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from nlp_processor import NLPAgent as NLPProcessorAgent
# from data_query import DataRetrievalAgent
# from kriging import InterpolatorAgent
# from image import MapRenderAgent
# from feedback_processor import FeedbackAgent as FeedbackParser
#
# # ------------------------
# # MCP 引入
# # ------------------------
# from mcp.server.fastmcp import Context, FastMCP
# from mcp.server.session import ServerSession
#
# mcp_server = FastMCP(name="Interpolation Pipeline")
#
# # ------------------------
# # Agent 抽象类
# # ------------------------
# class Agent:
#     async def run(self, ctx: Context[ServerSession, None], context: dict) -> dict:
#         raise NotImplementedError("Agent 必须实现 run 方法")
#
# # ------------------------
# # DummyContext，用于无 MCP 请求模式
# # ------------------------
# class DummyContext:
#     async def info(self, msg: str, **kwargs):
#         print(f"[INFO] {msg}")
#
#     async def report_progress(self, progress, total=1.0, message=""):
#         print(f"[PROGRESS] {progress*100:.1f}% - {message}")
#
# # ------------------------
# # Agent 实现
# # ------------------------
# class NLPAgentWrapper(Agent):
#     async def run(self, ctx: Context[ServerSession, None], context: dict) -> dict:
#         nlp_agent = NLPProcessorAgent()
#         context = await nlp_agent.run(ctx, context)
#
#         # 输出 NLP 解析结果到控制台
#         nlp_result = context.get("task") or context.get("nlp_result")
#         if nlp_result:
#             print("📝 NLP 解析结果:", nlp_result)
#             await ctx.info(f"📝 NLP 解析结果: {nlp_result}")
#         else:
#             print("⚠️ NLP 解析结果为空")
#             await ctx.info("⚠️ NLP 解析结果为空")
#
#         await ctx.info("🔹 NLP解析完成（通过异步 NLPAgent）")
#         return context
#
#
# class DataAgent(Agent):
#     async def run(self, ctx, context):
#         plan = context.get("plan", {})
#         pipeline = plan.get("pipeline")
#         if not pipeline:
#             await ctx.info("🔹 无 pipeline，跳过数据检索")
#             return context
#
#         step = pipeline[0]
#         params = step.get("params", {})
#
#         if context.get("data_points") and context.get("feedback"):
#             await ctx.info(f"🔹 复用历史数据 {len(context['data_points'])} 条")
#             return context
#
#         # 将 task_dict 放入 context，用于 DataRetrievalAgent
#         context_for_agent = {
#             "task": params
#         }
#
#         agent = DataRetrievalAgent()
#         context_result = await agent.run(ctx, context_for_agent)  # ✅ 异步调用
#
#         data_points = context_result.get("data_result", [])
#         context["data_points"] = data_points
#
#         await ctx.info(f"🔹 数据检索完成 {len(data_points)} 条")
#         return context
#
#
# class KrigingAgent(Agent):
#     FIELD_MAP = {"地层厚度": "thickness", "岩性": "lithology"}
#
#     def __init__(self):
#         self.interpolator_agent = InterpolatorAgent()
#
#     @staticmethod
#     def get_coord(point: dict, keys: list):
#         for k in keys:
#             if k in point:
#                 return point[k]
#         raise KeyError(f"None of {keys} found in {point}")
#
#     async def run(self, ctx: Context[ServerSession, None], context: dict) -> dict:
#         step = context["plan"]["pipeline"][1]
#         k_params = step.get("params", {})
#         data_points = context["data_points"]
#         task = context["task"]
#
#         variable_field = self.FIELD_MAP.get(task.variable, task.variable)
#
#         points = [
#             {
#                 "lon": self.interpolator_agent.get_coord(p, ["lon", "geo_X", "longitude"]),
#                 "lat": self.interpolator_agent.get_coord(p, ["lat", "geo_Y", "latitude"]),
#                 "value": self.interpolator_agent.get_coord(p, [variable_field, "value"])
#             }
#             for p in data_points
#         ]
#
#         task_dict = {
#             "method": k_params.get("method", "ok"),
#             "variogram_model": k_params.get("model", "spherical"),
#             "autoOptimizeModel": k_params.get("auto_optimize", False),
#             "smoothSigma": k_params.get("sigma", 0),
#             "drift": k_params.get("drift", "linear"),
#             "points": points
#         }
#
#         total_steps = len(points)
#         for i, point in enumerate(points):
#             await ctx.report_progress(progress=(i + 1)/total_steps, total=1.0,
#                                       message=f"插值处理中 {i+1}/{total_steps} 点")
#         kriging_result = await self.interpolator_agent.run(ctx, task_dict)
#
#         if "error" in kriging_result:
#             raise Exception(kriging_result["error"])
#
#         context["kriging_result"] = kriging_result
#         await ctx.info("🔹 插值计算完成")
#         return context
#
# class FeedbackAgent(Agent):
#     def __init__(self):
#         self.agent = FeedbackParser()
#
#     async def run(self, ctx: Context[ServerSession, None], context: dict) -> dict:
#         feedback_text = context.get("feedback")
#         if not feedback_text:
#             return context
#
#         mcp_context = self.agent.run(feedback_text)
#         feedback = mcp_context.params
#         context["feedbackParsed"] = feedback
#         await ctx.info(f"🔹 用户反馈解析: {feedback}")
#
#         if feedback.get("re_execute"):
#             context["re_execute"] = True
#             await ctx.info("🔹 用户要求重新执行插值")
#
#         return context
#
# # ------------------------
# # MCP Controller
# # ------------------------
# class MCPController:
#     def __init__(self):
#         self.last_context = {}
#         self.agents = {
#             "nlp": NLPAgentWrapper(),
#             "feedback": FeedbackAgent(),
#             "data": DataAgent(),
#             "kriging": KrigingAgent(),
#             "image": MapRenderAgent(),
#         }
#
#     async def run(self, context: dict, ctx=None) -> dict:
#         # 使用 DummyContext 兼容脚本模式
#         if ctx is None:
#             class DummyContext:
#                 async def info(self, msg: str, **kwargs):
#                     print(f"[INFO] {msg}")
#
#                 async def report_progress(self, progress, total=1.0, message=""):
#                     print(f"[PROGRESS] {progress*100:.1f}% - {message}")
#
#             ctx = DummyContext()
#
#         # === NLP 解析 ===
#         context = await self.agents["nlp"].run(ctx, context)
#         # === 反馈解析 ===
#         context = await self.agents["feedback"].run(ctx, context)
#         feedback = context.get("feedbackParsed") or context.get("feedback") or {}
#
#         # === 数据检索 ===
#         context = await self.agents["data"].run(ctx, context)
#         data_points = context.get("data_points", [])
#
#         # === 无新数据，使用历史结果 ===
#         if not data_points:
#             if self.last_context:
#                 await ctx.info("🔹 无新数据，使用历史 kriging_result + 用户反馈修改样式")
#                 # 应用用户反馈更新历史上下文
#                 for k, v in feedback.items():
#                     if v is not None:
#                         self.last_context[k] = v
#                 # 调用 MapRenderAgent 渲染
#                 return await self.agents["image"].run(ctx, self.last_context)
#             else:
#                 context.setdefault("errors", []).append("无历史结果，无法执行样式修改")
#                 return context
#
#         # === 根据反馈修改插值参数（如模型） ===
#         plan = context.setdefault("plan", {})
#         pipeline = plan.setdefault("pipeline", [])
#
#         if "model" in feedback and len(pipeline) > 1:
#             pipeline[1].setdefault("params", {})["model"] = feedback["model"]
#
#         # === 执行插值 ===
#         if len(pipeline) > 1:
#             context = await self.agents["kriging"].run(ctx, context)
#         else:
#             await ctx.info("🔹 无有效 pipeline，跳过 Kriging 插值")
#
#         # === 渲染图件 ===
#         context = await self.agents["image"].run(ctx, context)
#
#         # 保存历史上下文
#         self.last_context = context
#         return context
#
#
# # ------------------------
# # FastAPI
# # ------------------------
# app = FastAPI()
# mcp_controller = MCPController()
#
# class TaskRequest(BaseModel):
#     text: str
#     feedback: str | None = None
#
# def convert_to_json_serializable(obj):
#     if isinstance(obj, list):
#         return [convert_to_json_serializable(x) for x in obj]
#     elif isinstance(obj, dict):
#         return {k: convert_to_json_serializable(v) for k, v in obj.items()}
#     elif isinstance(obj, Decimal):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     else:
#         return obj
#
# @app.post("/task")
# async def run_task(req: TaskRequest):
#     try:
#         context = {
#             "text": req.text,
#             "feedback": req.feedback,
#             "task": mcp_controller.last_context.get("task"),
#             "plan": mcp_controller.last_context.get("plan"),
#             "data_points": mcp_controller.last_context.get("data_points"),
#         }
#
#         # MCP Context 直接传 None，MCPController 内部会自动处理
#         context = await mcp_controller.run(context=context, ctx=None)
#
#         return JSONResponse({
#             "nlpResult": context["task"].model_dump() if hasattr(context.get("task"), "model_dump") else context.get("task"),
#
#
#             # "nlpResult": context["task"].model_dump() if context.get("task") else None,
#             "plan": context.get("plan"),
#             "dataResult": convert_to_json_serializable(context.get("data_points")),
#             "krigingResult": convert_to_json_serializable(context.get("kriging_result")),
#             "imageResult": context.get("image_result"),
#             "geojsonResult": context.get("geojson_result"),
#             "feedbackUsed": req.feedback,
#             "feedbackParsed": context.get("feedbackParsed")
#         })
#
#     except Exception as e:
#         traceback.print_exc()
#         return JSONResponse(
#             status_code=500,
#             content={"error": str(e), "trace": traceback.format_exc()}
#         )
#
# # ------------------------
# # 脚本模式测试入口
# # ------------------------
# if __name__ == "__main__":
#     # import asyncio
#     # sample_context = {"text": "绘制四川盆地龙潭组灰岩分布图"}
#     # res = asyncio.run(mcp_controller.run(context=sample_context))
#     # print(">>> NLP 解析结果:", res.get("task"))
#     # print(">>> 插值/绘图流程完成")
#     #
#     # # 或启动 FastAPI
#     import uvicorn
#     uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from decimal import Decimal
import math
import traceback
import json
import hashlib

# ------------------------
# MCP 引入
# ------------------------
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession
from mcp_server import mcp_server  # ✅ 直接导入正确的 mcp_server 实例
from mcp_tool import *
from agent import NLPAgent, DataAgent, KrigingAgent, MapRenderAgent, FeedbackAgent, OverlayAgent, ExtendedContext
from context_schema import MCPContextSchema # 导入 Schema 用于重置

# ------------------------
# FastAPI 初始化
# ------------------------
app = FastAPI()

# ------------------------
# MCP Controller + 调度
# ------------------------
class MCPController:
    def __init__(self):
        self.last_context = {}
        self.agents = {
            "nlp": NLPAgent(),
            "data": DataAgent(),
            "kriging": KrigingAgent(),
            "overlay": OverlayAgent(),
            "image": MapRenderAgent(),
            "feedback": FeedbackAgent(),
        }
        self.history = []  # 历史记录列表

    async def run_pipeline(self, context, ctx):
        extended_ctx = ExtendedContext.from_context(ctx)

        # --- 步骤 1: 判断运行模式 (新任务 vs 反馈) ---
        is_feedback_run = context.get("is_feedback_run", False)

        if is_feedback_run:
            # 反馈模式: 先运行 FeedbackAgent，它会处理缓存失效
            context = await self.agents["feedback"].run(extended_ctx, context)
        else:
            # --- 新任务模式 ---
            # 1. 运行 NLPAgent 解析用户意图
            context = await self.agents["nlp"].run(extended_ctx, context)
            
            # 2. ✨ 基于 NLP 解析结果 (任务意图) 进行缓存检查 ✨
            task_params = context.get("task")
            if task_params:
                # 将任务参数字典转换为稳定的、排序后的 JSON 字符串
                params_str = json.dumps(task_params, sort_keys=True, ensure_ascii=False)
                # 使用哈希生成唯一的缓存键
                cache_key = f"task::{hashlib.md5(params_str.encode('utf-8')).hexdigest()}"

                if cache_key in mcp_server.context.run_cache:
                    await extended_ctx.info(f"✅ 任务意图缓存命中 (Key: {cache_key})，直接返回最终结果。")
                    cached_context = mcp_server.context.run_cache[cache_key]
                    # 为了保持一致性，只返回必要的核心结果
                    return {
                        "task": cached_context.get("task"),
                        "plan": {"pipeline": []}, # 表示流程已完成
                        "data_points_by_variable": cached_context.get("data", {}).get("data_points_by_variable"),
                        "kriging_result": cached_context.get("results", {}).get("kriging"),
                        "overlay_result": cached_context.get("results", {}).get("overlay"),
                        "render_results": cached_context.get("results", {}).get("render"),
                        "feedbackParsed": None,
                        "cached": True,
                    }
                else:
                    await extended_ctx.info(f"ℹ️ 任务意图无缓存 (Key: {cache_key})，继续执行流程。")
                    # 将缓存键存入 context，以便流程结束后写入缓存
                    context["task_cache_key"] = cache_key

        # --- 步骤 2: 动态执行 Pipeline ---
        pipeline = context.get("plan", {}).get("pipeline", [])
        if not pipeline:
            await extended_ctx.error("未能生成或继承有效的执行计划 (pipeline)")
            context.setdefault("errors", []).append("未能生成执行计划")
            return context

        await extended_ctx.info(f"动态执行计划: {' -> '.join(pipeline)}")

        for agent_name in pipeline:
            # 在反馈模式下，nlp agent 不应再执行
            if is_feedback_run and agent_name == "nlp":
                continue
            
            agent = self.agents.get(agent_name)
            if agent:
                context = await agent.run(extended_ctx, context)
            else:
                await extended_ctx.error(f"未找到名为 '{agent_name}' 的 Agent")

        self.last_context = context

        # 保存历史记录（包括参数）
        history_entry = {
            "text": context.get("text"),
            "feedback": context.get("feedback"),
            "params": {
                "kriging": context.get("kriging_params", {}),
                "render": context.get("render_params", {}),
            },
            "dataResult": convert_to_json_serializable(context.get("data_points")),
            "krigingResult": convert_to_json_serializable(context.get("kriging_result")),
            "imageResult": convert_to_json_serializable(context.get("image_results")),
            "geojsonResult": convert_to_json_serializable(context.get("geojson_results")),
            "timestamp": datetime.now().isoformat()
        }
        self.history.append(history_entry)

        # 限制历史长度，防止内存过大
        if len(self.history) > 20:
            self.history.pop(0)

        # --- 流程成功完成，将最终结果写入“任务意图”缓存 ---
        task_cache_key = context.get("task_cache_key")
        if task_cache_key and not context.get("errors"):
            import copy
            # 缓存当前完整的 MCPContext 状态
            mcp_server.context.run_cache[task_cache_key] = {
                "task": copy.deepcopy(mcp_server.context.task),
                "params": copy.deepcopy(mcp_server.context.params),
                "data": copy.deepcopy(mcp_server.context.data),
                "results": copy.deepcopy(mcp_server.context.results),
            }
            await extended_ctx.info(f"✅ 流程成功，最终结果已写入任务缓存 (Key: {task_cache_key})")

        return context


mcp_controller = MCPController()

# ------------------------
# 请求模型
# ------------------------
class TaskRequest(BaseModel):
    text: str | None = None
    feedback: str | None = None  # 用户反馈是字符串

# ------------------------
# 工具函数：JSON 可序列化转换（处理 NaN / Inf / Decimal / np.ndarray）
# ------------------------
def convert_to_json_serializable(obj):
    if isinstance(obj, list):
        return [convert_to_json_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return convert_to_json_serializable(obj.tolist())
    else:
        return obj

# ------------------------
# MCP 任务执行
# ------------------------
async def run_task_mcp(text: str | None = None, feedback: str | None = None) -> dict:
    ctx = Context[ServerSession, None](session=None)

    # --- 启发式规则：检测可能错发到 'text' 字段的反馈指令 ---
    if not feedback and text:
        feedback_keywords = ["修改", "更改", "换成", "渲染", "颜色",  "色带"]
        # 假设反馈指令通常较短，且包含关键词
        if any(keyword in text for keyword in feedback_keywords) and len(text.split()) < 15:
            feedback = text  # 将 text 内容视为 feedback
            text = None      # 清空 text，强制进入反馈模式

    # --- 判断是新任务还是反馈 ---
    if feedback:
        # 反馈模式
        if not mcp_controller.last_context:
            return {"error": "No previous task context available to apply feedback."}
        
        # 继承上一次的上下文，并加入新的反馈
        context = mcp_controller.last_context.copy()
        context["feedback"] = feedback
        context["is_feedback_run"] = True
        
    else:
        # 新任务模式
        print("✨ New task detected, applying soft reset to MCPContext.")
        
        # 软重置：仅保留 run_cache，清空其他所有内容，确保新任务使用默认参数
        existing_run_cache = {} # 默认为空字典

        if hasattr(mcp_server, 'context') and mcp_server.context:
            # 关键：只保留现有的 run_cache
            existing_run_cache = mcp_server.context.run_cache

        # 执行重置：创建一个全新的上下文对象
        mcp_server.context = MCPContextSchema()
        
        # 仅恢复 run_cache
        mcp_server.context.run_cache = existing_run_cache
        
        if existing_run_cache:
            print(f"📦 Kept {len(existing_run_cache)} items in run_cache.")
        
        if existing_run_cache:
            print(f"📦 Kept {len(existing_run_cache)} items in run_cache.")

        context = {
            "text": text,
            "feedback": None,
            "is_feedback_run": False,
        }
        
    return await mcp_controller.run_pipeline(context, ctx=ctx)

# ------------------------
# FastAPI 接口
# ------------------------
@app.post("/task")
async def run_task(req: TaskRequest):
    try:
        result_context = await run_task_mcp(req.text, req.feedback)

        render_results = result_context.get("render_results", {})

        # --- 合并单因素和多因素的数据点 ---
        merged_data_points = []
        data_by_variable = result_context.get("data_points_by_variable", {})
        if data_by_variable:
            for variable, points in data_by_variable.items():
                for point in points:
                    new_point = point.copy()
                    new_point['variable'] = variable
                    merged_data_points.append(new_point)
        
        response_content = {
            "nlpResult": result_context.get("task"),
            "plan": result_context.get("plan"),
            "dataResult": convert_to_json_serializable(merged_data_points),
            "krigingResult": convert_to_json_serializable(result_context.get("kriging_result")),
            "imageResult": render_results.get("image_base64"),
            "geojsonResult": render_results.get("geojson"),
            "feedbackParsed": result_context.get("feedbackParsed"),
            "history": mcp_controller.history
        }

        return JSONResponse(response_content)
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback_str}
        )


# ------------------------
# FastAPI 接口：获取历史记录
# ------------------------
@app.get("/history")
async def get_history(limit: int = Query(20, ge=1)):
    """
    返回最近 limit 条历史记录
    """
    try:
        # 截取最近 limit 条
        history_slice = mcp_controller.history[-limit:]
        return JSONResponse({"history": history_slice})
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback_str}
        )

# ------------------------
# 启动入口
# ------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="10.242.48.50", port=8000, reload=True)

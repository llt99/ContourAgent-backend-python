import asyncio
from collections import defaultdict
import logging
import json
import hashlib

import numpy as np

from kriging import Interpolator
from mcp_server import mcp_server
from pprint import pformat
from prompt import SYSTEM_TO_STRATA
from ahp_calculator import calculate_ahp_weights
# ------------------------
# 抽象 Agent
# ------------------------
class Agent:
    async def run(self, ctx, context: dict) -> dict:
        """必须实现 run 方法"""
        raise NotImplementedError("Agent 必须实现 run 方法")

# ------------------------
# DummyContext，用于测试或无 MCP 模式
# ------------------------
class DummyContext:
    async def info(self, msg: str, **kwargs):
        logging.info(msg)

    async def report_progress(self, progress, total=1.0, message=""):
        logging.info(f"[PROGRESS] {progress*100:.1f}% - {message}")

    async def error(self, msg: str, **kwargs):
        logging.error(msg)

class ExtendedContext:
    def __init__(self, ctx=None):
        self.ctx = ctx

    @classmethod
    def from_context(cls, ctx):
        return cls(ctx)

    async def info(self, msg: str):
        if hasattr(self.ctx, "info"):
            try:
                await self.ctx.info(msg)
                return
            except Exception:
                pass
        logging.info(msg)

    async def error(self, msg: str):
        if hasattr(self.ctx, "error"):
            try:
                await self.ctx.error(msg)
                return
            except Exception:
                pass
        logging.error(msg)

    async def report_progress(self, progress: float, total: float = 1.0, message: str = ""):
        if hasattr(self.ctx, "report_progress"):
            try:
                await self.ctx.report_progress(progress, total, message)
                return
            except Exception:
                pass
        logging.info(f"[PROGRESS] {progress*100:.1f}% {message}")

    async def call_tool(self, name: str, **kwargs):
        """
        优先使用 ctx.call_tool，否则 fallback 到全局 mcp_server。
        内置精细化缓存逻辑。
        """
        # --- 1. 生成缓存键 ---
        # 移除 ctx 参数，因为它会变化且不影响结果
        kwargs_for_key = {k: v for k, v in kwargs.items() if k != 'ctx'}

        # 定义一个转换函数来处理 Decimal 和其他非序列化类型
        def json_converter(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if hasattr(o, 'isoformat'): # 处理 datetime 对象
                return o.isoformat()
            # 关键：处理 Decimal 类型
            from decimal import Decimal
            if isinstance(o, Decimal):
                return float(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        # 将参数字典转换为稳定的、排序后的 JSON 字符串
        try:
            params_str = json.dumps(kwargs_for_key, sort_keys=True, ensure_ascii=False, default=json_converter)
        except TypeError as e:
            # 提供更详细的错误日志，帮助定位问题
            logging.error(f"无法序列化工具 '{name}' 的参数用于生成缓存键: {e}")
            # 为了避免程序中断，可以选择不使用缓存继续执行
            # 这里我们选择重新抛出，因为缓存是核心功能
            raise e
            
        # 使用 MD5 生成简短的哈希值作为键
        cache_key_hash = hashlib.md5(params_str.encode('utf-8')).hexdigest()
        cache_key = f"{name}::{cache_key_hash}"

        # --- 2. 检查缓存 ---
        if cache_key in mcp_server.context.run_cache:
            await self.info(f"✅ 工具 '{name}' 缓存命中 (Key: ...{cache_key_hash[-6:]})")
            return mcp_server.context.run_cache[cache_key]

        await self.info(f"🚀 执行工具 '{name}' (无缓存, Key: ...{cache_key_hash[-6:]})")

        # --- 3. 执行工具 ---
        # 确保 ctx 参数始终传入
        if "ctx" not in kwargs:
            kwargs["ctx"] = self  # self 是 ExtendedContext

        result = None
        try:
            # 先尝试 MCP Context 内调用
            if hasattr(self, "_ctx") and hasattr(self._ctx, "call_tool"):
                try:
                    result = await self._ctx.call_tool(name, **kwargs)
                except Exception:
                    logging.warning(f"⚠️ MCP Context 调用 {name} 失败，尝试全局 mcp_server")
                    result = None # 确保在失败时 result 为 None

            # 如果上一步没成功，则 fallback 全局 mcp_server
            if result is None:
                tool = mcp_server._local_tools.get(name)
                if not tool:
                    raise RuntimeError(f"工具 {name} 未注册")

                if asyncio.iscoroutinefunction(tool):
                    result = await tool(**kwargs)
                else:
                    result = tool(**kwargs)

        except Exception as e:
            await self.error(f"❌ 工具 '{name}' 执行失败: {e}")
            raise  # 重新抛出异常，让上层逻辑处理

        # --- 4. 写入缓存 ---
        if result is not None:
            mcp_server.context.run_cache[cache_key] = result
            await self.info(f"📝 工具 '{name}' 结果已写入缓存")

        return result


# ------------------------
# NLP Agent
# ------------------------
class NLPAgent(Agent):
    async def run(self, ctx: ExtendedContext, context: dict) -> dict:
        text = context.get("text")
        if not text:
            context.setdefault("errors", []).append("缺少 text")
            context["plan"] = {"pipeline": []}
            return context

        try:
            # 1. 调用 NLP 工具解析
            nlp_result = await ctx.call_tool(
                "parse_text_tool_mcp",
                user_text=text,
                context=context
            )
            task = nlp_result.get("task", {})
            plan = nlp_result.get("plan", {"pipeline": []})

            # 2. 动态调整 plan 以加入 OverlayAgent
            analysis_type = task.get("analysis_type")
            if analysis_type == "multi_factor":
                pipeline = plan.get("pipeline", [])
                try:
                    kriging_index = pipeline.index("kriging")
                    # 在 kriging 之后，image 之前插入 overlay
                    if "overlay" not in pipeline:
                        pipeline.insert(kriging_index + 1, "overlay")
                    plan["pipeline"] = pipeline
                    await ctx.info("✅ 检测到多因素分析任务，已在流程中加入 OverlayAgent")
                except ValueError:
                    await ctx.warning("⚠️ 多因素任务的 plan 中未找到 'kriging'，无法自动插入 'overlay'")
            
            context["task"] = task
            context["plan"] = plan

            # 3. 参数继承与合并
            last_params = getattr(mcp_server, "context", {}).params or {}
            context["params"] = {**last_params, **task}

            # 4. 写回 MCPContext
            mcp_server.context.task.update(task)
            mcp_server.context.params.update(context["params"])
            mcp_server.context.task["plan"] = plan

            # 5. 输出状态
            await ctx.info(" NLPAgent 执行后，全局 MCPContext 状态：")
            await ctx.info(pformat({
                "task": mcp_server.context.task,
                "params": mcp_server.context.params,
                "plan": plan
            }, width=80))

        except Exception as e:
            context.setdefault("errors", []).append(str(e))
            await ctx.error(f"NLP 解析失败: {e}")

        return context


# ------------------------
# Feedback Agent
# ------------------------
class FeedbackAgent(Agent):
    async def run(self, ctx: ExtendedContext, context: dict) -> dict:
        feedback_text = context.get("feedback")
        if not feedback_text:
            await ctx.info("⚠️ 无用户反馈，跳过 FeedbackAgent")
            return context

        # --- 缓存失效逻辑 ---
        # 当有用户反馈时，意味着之前的整个流程可能都需要重新计算。
        # 最稳妥的办法是清空所有工具缓存。
        original_text = context.get("text")
        if original_text:
            # 清除顶层任务缓存
            if original_text in mcp_server.context.run_cache:
                del mcp_server.context.run_cache[original_text]
                await ctx.info(f"ℹ️ 用户反馈已使顶层任务 '{original_text}' 的缓存失效")
            
            # 清除所有精细化工具缓存
            keys_to_del = [k for k in mcp_server.context.run_cache.keys() if "::" in k]
            if keys_to_del:
                for k in keys_to_del:
                    del mcp_server.context.run_cache[k]
                await ctx.info(f"ℹ️ 用户反馈导致 {len(keys_to_del)} 个工具缓存被清除")
        else:
             await ctx.warning("⚠️ 无法确定反馈对应的原始任务，缓存可能未完全失效")

        try:
            # 修正：确保所有参数都通过关键字传递，避免位置参数冲突
            result = await ctx.call_tool(
                "parse_user_feedback_tool",
                feedback_text=feedback_text,
                context=context
            )
            
            # 更新 MCP 上下文中的核心参数
            if "params" in result.get("mcp_context", {}):
                mcp_server.context.params.update(result["mcp_context"]["params"])

            # 将解析出的参数也更新到当前任务的本地上下文中
            # 确保 context 中有 params 字典
            if "params" not in context:
                context["params"] = {}
            context["params"].update(result["mcp_context"]["params"])
            await ctx.info("✅ 用户反馈已更新到 MCPContext")
        except Exception as e:
            context.setdefault("errors", []).append(str(e))
            mcp_server.context.add_error(str(e))
            await ctx.error(f"反馈解析失败: {e}")
        return context



# ------------------------
# Data Agent
# ------------------------
class DataAgent(Agent):
    async def run(self, ctx: ExtendedContext, context: dict) -> dict:
        task = context.get("task")
        if not task:
            await ctx.error("缺少 task，DataAgent 无法执行")
            return context

        # ---- 兼容单/多因素 ----
        variables = task.get("variables")
        if not variables:
            # 兼容旧的单因素模式
            variable = task.get("variable")
            if not variable:
                await ctx.error("任务中缺少 'variables' 或 'variable' 字段")
                return context
            variables = [variable]
        
        await ctx.info(f"🔍 DataAgent 开始为 {len(variables)} 个变量获取数据: {variables}")

        data_points_by_variable = {}
        stratum = task.get("stratum")
        system = task.get("system")

        for variable in variables:
            await ctx.info(f"--- 正在查询变量: {variable} ---")
            
            # ---- 构造查询文本 ----
            if system and not stratum:
                query_text = f"查询 {system} 各井 {variable} 数据（包含全部子地层）"
            else:
                query_text = f"查询 {stratum} 各井 {variable} 数据"
            
            await ctx.info(f"🧭 数据检索任务: {query_text}")

            # ---- 执行 SQL 查询 ----
            try:
                query_result = await ctx.call_tool("text_to_sql_query_tool", query=query_text)
                if not query_result or not query_result.get("rows"):
                    await ctx.warning(f"⚠️ 变量 '{variable}' 未获取到有效井点数据")
                    continue
                
                await ctx.info(f"✅ 变量 '{variable}' 获取到 {len(query_result['rows'])} 个井点数据")
                data_points_by_variable[variable] = query_result["rows"]

            except Exception as e:
                context.setdefault("errors", []).append(f"变量 '{variable}' 查询失败: {str(e)}")
                await ctx.error(f"❌ 变量 '{variable}' 数据查询失败: {e}")
                continue
        
        if not data_points_by_variable:
            await ctx.error("❌ 所有变量均未能获取到数据，无法继续")
            return context

        # ---- 将结果写入上下文 ----
        context["data_points_by_variable"] = data_points_by_variable
        mcp_server.context.data["data_points_by_variable"] = data_points_by_variable
        
        # ---- 兼容旧模式：如果只有一个变量，则填充旧的 data_points 字段 ----
        if len(variables) == 1:
            single_variable = variables[0]
            if single_variable in data_points_by_variable:
                context["data_points"] = {"rows": data_points_by_variable[single_variable]}
                mcp_server.context.data["data_points"] = data_points_by_variable[single_variable]

        await ctx.info("✅ 所有变量数据已获取并写入 MCPContext")
        return context


# ------------------------
# Kriging Agent
# ------------------------
class KrigingAgent(Agent):
    async def run(self, ctx: ExtendedContext, context: dict) -> dict:
        # ---- 兼容新旧数据结构 ----
        data_points_by_variable = context.get("data_points_by_variable")
        if not data_points_by_variable:
            # 兼容旧模式
            rows = context.get("data_points", {}).get("rows", [])
            if not rows:
                await ctx.info("⚠️ 无数据点，跳过 KrigingAgent")
                return context
            # 将旧结构包装成新结构
            variable = context.get("task", {}).get("variable", "unknown_variable")
            data_points_by_variable = {variable: rows}
            await ctx.info("🔄 检测到旧版单因素数据结构，已自动兼容")

        task = context.get("task", {})
        kriging_results = {}

        await ctx.info(f"🚀 KrigingAgent 开始为 {len(data_points_by_variable)} 个变量执行插值...")

        for variable, rows in data_points_by_variable.items():
            await ctx.info(f"--- 正在插值变量: {variable} ---")
            
            # --- 增加对岩相古地理这类分类数据的特殊处理 ---
            if variable == "岩相古地理":
                await ctx.info(f"ℹ️ 检测到分类数据 '{variable}'，将跳过数值插值，直接进行点渲染。")
                kriging_results[variable] = {
                    "grid_x": None, "grid_y": None, "z": None,
                    "is_categorical_points": True,
                    "points": rows
                }
                # 这里是关键修复：我们应该直接 continue，完全跳过后续的工具调用
                continue

            if len(rows) < 5:
                await ctx.error(f"❌ 变量 '{variable}' 数据点过少 ({len(rows)} 个)，无法执行插值")
                kriging_results[variable] = {"error": "数据点不足"}
                continue

            # ---- 数据点格式化 ----
            points = []
            for p in rows:
                lon = p.get("lon") or p.get("geo_X")
                lat = p.get("lat") or p.get("geo_Y")
                value = p.get("value") or p.get("thickness") or p.get("ratio") or p.get("content")
                if None in (lon, lat, value):
                    continue
                points.append({"lon": float(lon), "lat": float(lat), "value": float(value)})

            # 只有在 points 列表不为空时才执行插值
            if not points:
                await ctx.warning(f"⚠️ 变量 '{variable}' 没有有效的数值点可供插值，已跳过。")
                kriging_results[variable] = {"error": "无有效数值点"}
                continue

            try:
                # ---- 参数提取 ----
                params = mcp_server.context.params
                method_raw = params.get("method") or task.get("method_code") or "auto"
                model_raw = params.get("variogram_model") or task.get("model_code") or "auto"
                method_map = {"普通克里金": "ok", "泛克里金": "uk", "universal_kriging": "uk", "ordinary_kriging": "ok", "反距离加权": "idw", "idw": "idw"}
                method = method_map.get(str(method_raw).lower(), str(method_raw).lower())
                candidate_models = params.get("candidate_models", ["spherical", "exponential", "gaussian"])
                auto_optimize = params.get("auto_optimize", True)
                drift = params.get("drift", "linear")
                idw_power = params.get("idw_power", 2.0)

                if model_raw and model_raw != "auto":
                    candidate_models = [model_raw]
                    auto_optimize = False
                    await ctx.info(f"🎯 用户为变量 '{variable}' 指定半变异函数模型: {model_raw}")

                await ctx.info(f"⚙️ 变量 '{variable}' 插值参数: method={method}, models={candidate_models}, drift={drift}")

                # ---- 执行插值 ----
                interp_result = await ctx.call_tool(
                    "kriging_interpolate",
                    points=points,
                    method=method,
                    candidate_models=candidate_models,
                    autoOptimizeModel=auto_optimize,
                    drift=drift,
                    idw_power=idw_power,
                )

                if not interp_result or "error" in interp_result:
                    raise Exception(interp_result.get("error", "插值返回空结果"))

                kriging_results[variable] = interp_result
                await ctx.info(f"✅ 变量 '{variable}' 插值完成")

            except Exception as e:
                await ctx.error(f"❌ 变量 '{variable}' 插值失败: {e}")
                kriging_results[variable] = {"error": str(e)}

        # ---- 将结果写入上下文 ----
        context["kriging_result"] = kriging_results
        mcp_server.context.results["kriging"] = kriging_results
        
        await ctx.info("✅ 所有变量插值完成，结果已写入 MCPContext")
        return context


# ------------------------
# Overlay Agent
# ------------------------
class OverlayAgent(Agent):
    # 定义沉积相分类规则
    # 规则格式: (沉积相名称, 颜色, lambda函数)
    # lambda 函数接收一个包含所有变量值的字典，返回 True 或 False
    SEDIMENTARY_FACIES_RULES = [

        # ==========================================================
        # 1. 火山环境（玄武岩为主，属于高能环境）
        # ==========================================================
        ("火山环境", "#8B0000", lambda v: (
                v.get("玄武岩", 0) >= 0.50 and  # 玄武岩含量大
                v.get("煤岩", 0) < 0.05 and  # 煤岩几乎没有
                v.get("碳酸盐岩", 0) < 0.10  # 碳酸盐岩少
        )),

        # ==========================================================
        # 2. 碳酸盐岩环境（浅海或浅湖环境）
        # ==========================================================
        ("碳酸盐岩环境", "#32CD32", lambda v: (
                v.get("碳酸盐岩", 0) >= 0.50 and  # 碳酸盐岩含量大
                v.get("碎屑岩", 0) < 0.20 and  # 碎屑岩少
                v.get("煤岩", 0) < 0.05  # 煤岩几乎没有
        )),

        # ==========================================================
        # 3. 三角洲环境（碎屑岩和煤岩含量较高）
        # ==========================================================
        ("三角洲环境", "#FFD700", lambda v: (
                v.get("碎屑岩", 0) >= 0.50 and  # 碎屑岩占主导
                v.get("煤岩", 0) <= 0.30 and  # 煤岩次要
                v.get("碳酸盐岩", 0) < 0.10  # 碳酸盐岩少
        )),

        # ==========================================================
        # 4. 泻湖环境（膏盐岩为主，少量煤岩）
        # ==========================================================
        ("泻湖环境", "#00FFFF", lambda v: (
                v.get("膏盐岩", 0) >= 0.50 and  # 膏盐岩占主导
                v.get("煤岩", 0) <= 0.10 and  # 煤岩少
                v.get("碎屑岩", 0) < 0.20  # 碎屑岩少
        )),

        # ==========================================================
        # 5. 沼泽环境（煤岩为主）
        # ==========================================================
        ("沼泽环境", "#2F4F4F", lambda v: (
                v.get("煤岩", 0) >= 0.50 and  # 煤岩占主导
                v.get("碎屑岩", 0) < 0.20 and  # 碎屑岩少
                v.get("膏盐岩", 0) < 0.10  # 膏盐岩少
        )),

        # ==========================================================
        # 6. 硅质沉积环境（硅岩为主）
        # ==========================================================
        ("硅质沉积环境", "#4682B4", lambda v: (
                v.get("硅岩", 0) >= 0.50 and  # 硅岩占主导
                v.get("煤岩", 0) < 0.10 and  # 煤岩少
                v.get("碎屑岩", 0) < 0.20  # 碎屑岩少
        )),

        # ==========================================================
        # 7. 未分类（没有明确岩性特征时使用）
        # ==========================================================
        ("未分类", "#D3D3D3", lambda v: True)  # 默认未分类
    ]

    DEFAULT_FACIES = ("未分类", "#D3D3D3")

    async def run(self, ctx: ExtendedContext, context: dict) -> dict:
        kriging_results = context.get("kriging_result", {})
        valid_factors = [f for f, r in kriging_results.items() if "error" not in r and "z" in r]

        if len(valid_factors) < 2:
            await ctx.info("⚠️ 有效插值结果不足两个，跳过沉积相分析")
            if len(valid_factors) == 1:
                context["task"]["variable"] = valid_factors[0]
            return context

        await ctx.info(f"🚀 开始基于规则的沉积相分析，涉及变量: {valid_factors}")

        # ---- 1. 变量名标准化 ----
        # 创建一个从原始变量名到标准化名称（如“泥岩”）的映射
        key_mapping = {}
        for factor in valid_factors:
            # 移除常见的后缀
            clean_factor = factor.replace("厚度", "").replace("含量", "")
            key_mapping[factor] = clean_factor
        await ctx.info(f"ℹ️ 标准化后变量名映射: {key_mapping}")

        # ---- 2. 提取所有插值网格数据 (并确保是 numpy array) ----
        grids = {factor: np.array(kriging_results[factor]["z"]) for factor in valid_factors}
        
        # 检查网格形状是否一致
        first_shape = next(iter(grids.values())).shape
        if not all(grid.shape == first_shape for grid in grids.values()):
            await ctx.error("❌ 各变量插值网格形状不一致，无法进行分类")
            return context
        
        grid_shape = first_shape

        # ---- 3. 数据归一化：将厚度转换为百分比 ----
        await ctx.info("⚖️ 开始进行数据归一化（厚度 -> 百分比）")
        total_thickness = np.zeros(grid_shape)
        for factor in valid_factors:
            # 将负值或 NaN 值视为 0，避免影响总厚度计算
            total_thickness += np.nan_to_num(grids[factor], nan=0.0, neginf=0.0, posinf=0.0)

        # 避免除以零
        total_thickness[total_thickness == 0] = 1.0

        normalized_grids = {factor: grids[factor] / total_thickness for factor in valid_factors}
        await ctx.info("✅ 数据归一化完成")

        facies_grid = np.full(grid_shape, -1, dtype=int) # -1 代表未分类

        # ---- 4. 增加调试日志：打印第一个有效点的数据 ----
        logged = False
        for r_idx in range(grid_shape[0]):
            for c_idx in range(grid_shape[1]):
                if not logged and total_thickness[r_idx, c_idx] > 1.0: # 找一个有实际厚度的点
                    raw_values = {f: grids[f][r_idx, c_idx] for f in valid_factors}
                    norm_values = {key_mapping[f]: normalized_grids[f][r_idx, c_idx] for f in valid_factors}
                    await ctx.info("---- 🔍 调试日志：第一个有效点数据 ----")
                    await ctx.info(f"坐标: ({r_idx}, {c_idx})")
                    await ctx.info(f"原始厚度值: {raw_values}")
                    await ctx.info(f"计算出的总厚度: {total_thickness[r_idx, c_idx]}")
                    await ctx.info(f"归一化后的百分比: {norm_values}")
                    await ctx.info("------------------------------------")
                    logged = True
                    break
            if logged:
                break

        # ---- 5. 逐点应用规则进行分类 ----
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                # 使用标准化后的变量名构建用于规则判断的字典
                values_at_point = {key_mapping[factor]: normalized_grids[factor][i, j] for factor in valid_factors}
                
                # 跳过无效点
                if any(np.isnan(v) for v in values_at_point.values()):
                    continue

                # 应用规则
                classified = False
                for idx, (name, color, rule_func) in enumerate(self.SEDIMENTARY_FACIES_RULES):
                    if rule_func(values_at_point):
                        facies_grid[i, j] = idx
                        classified = True
                        break # 应用第一个匹配的规则
                
                if not classified:
                    facies_grid[i, j] = len(self.SEDIMENTARY_FACIES_RULES) # 默认分类的索引

        # ---- 6. 准备分类结果用于渲染 ----
        # 获取 grid_x, grid_y
        grid_x = kriging_results[valid_factors[0]]["grid_x"]
        grid_y = kriging_results[valid_factors[0]]["grid_y"]

        # 创建颜色映射和标签
        facies_names = [name for name, _, _ in self.SEDIMENTARY_FACIES_RULES] + [self.DEFAULT_FACIES[0]]
        facies_colors = [color for _, color, _ in self.SEDIMENTARY_FACIES_RULES] + [self.DEFAULT_FACIES[1]]
        
        # 将分类结果打包，以兼容后续的 MapRenderAgent
        # 注意：我们将分类网格（整数）放入 'z'，并提供分类信息
        overlay_result = {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "z": facies_grid,
            "is_categorical": True, # 标记为分类数据
            "categories": {
                "names": facies_names,
                "colors": facies_colors
            }
        }

        # ---- 7. 将结果写入上下文 ----
        context["overlay_result"] = overlay_result
        mcp_server.context.results["overlay"] = overlay_result
        
        # 替换 kriging_result 中的第一个结果为我们的分类结果
        # 这使得后续的 MapRenderAgent 可以直接使用这个分类结果进行渲染
        first_key = valid_factors[0]
        context["kriging_result"][first_key] = overlay_result
        context["task"]["variable"] = "沉积相分布" # 更新任务变量名，以便图例显示正确

        await ctx.info("✅ 沉积相分类完成，结果已生成并传递给渲染模块")
        return context


# ------------------------
# MapRender Agent
# ------------------------
class MapRenderAgent(Agent):
    async def run(self, ctx: ExtendedContext, context: dict) -> dict:

        # await ctx.info(f"🎯 绘图前 MCPContext params: {mcp_server.context.params}")

        kriging_results = context.get("kriging_result") or mcp_server.context.results
        if not kriging_results:
            await ctx.info("⚠️ 无插值结果，跳过 MapRenderAgent")
            return context

        # 在多因素分析后，kriging_results 的第一个元素已被替换为 overlay_result
        first_result = next(iter(kriging_results.values()))
        
        # --- 对岩相古地理这类特殊分类数据进行处理 ---
        if first_result.get("is_categorical_points"):
            await ctx.info("ℹ️ 渲染分类点数据...")
            # 对于分类点数据，我们直接调用渲染工具，但不传递网格数据
            res = await ctx.call_tool(
                "render_map_tool",
                grid_x=None,
                grid_y=None,
                z=None,
                first_result=first_result,
                points=first_result.get("points", []),
                variable=context.get("task", {}).get("variable"),
            )
            context["render_results"] = {
                "image_base64": res.get("image_base64"),
                "geojson": res.get("geojson"),
                "colors": res.get("colors", [])
            }
            await ctx.info("✅ 分类点数据渲染完成")
            return context

        try:
            # --- 合并所有变量的数据点用于渲染 ---
            all_points = []
            if context.get("data_points_by_variable"):
                for points_list in context["data_points_by_variable"].values():
                    all_points.extend(points_list)
            else:
                # 兼容单因素模式
                all_points = context.get("data_points", {}).get("rows", [])

            params = mcp_server.context.params
            res = await ctx.call_tool(
                "render_map_tool",
                grid_x=first_result["grid_x"],
                grid_y=first_result["grid_y"],
                z=first_result["z"],
                first_result=first_result,  # 传递完整结果以区分是分类数据还是连续数据
                points=all_points,
                variable=context.get("task", {}).get("variable"),
                colormap=params.get("colormap", "RdYlBu"),
                n_classes=params.get("n_classes"),
                smooth_sigma=params.get("smooth_sigma", 0),
                lighten=params.get("lighten", False)
            )
            # ---- 将结果和参数写入本地和全局上下文 ----
            render_results = {
                "image_base64": res.get("image_base64"),
                "geojson": res.get("geojson"),
                "colors": res.get("colors", [])
            }
            
            context["render_results"] = render_results
            mcp_server.context.results["render"] = render_results

            # ---- 回写最终使用的渲染参数到全局上下文 ----
            final_render_params = {
                "colormap": params.get("colormap", "RdYlBu"),
                "n_classes": params.get("n_classes"),
                "smooth_sigma": params.get("smooth_sigma", 0),
                "lighten": params.get("lighten", False)
            }
            mcp_server.context.params.update(final_render_params)

            await ctx.info("✅ 渲染结果和最终使用参数已写入 MCPContext")
        except Exception as e:
            await ctx.error(f"❌ 地图渲染失败: {e}")
            context.setdefault("errors", []).append(str(e))
            mcp_server.context.add_error(str(e))

        return context

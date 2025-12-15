import numpy as np
import asyncio
from typing import Dict, Any, List, Optional
from mcp_server import mcp_server
from data_query import text_to_sql_query
from nlp_processor import parse_text_tool
from kriging import Interpolator
from image import MapRenderer
from feedback_processor import parse_user_feedback
from functools import wraps
import inspect

def mcp_tool_history(history_key: str, extract_input=None):
    """
    装饰器：自动管理 MCP 工具历史上下文
    - history_key: ctx.memory 中的历史记录 key
    - extract_input: 函数，提取输入数据存入历史
    自动记录 mcp_server.context.params 当前状态
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(ctx, *args, **kwargs):
                if not hasattr(ctx, "memory"):
                    ctx.memory = {}
                ctx.memory.setdefault(history_key, [])

                # 提取工具输入
                input_data = extract_input(*((ctx,) + args), **kwargs) if extract_input else {"args": args, "kwargs": kwargs}

                # 记录当前 MCPContext.params
                input_data["context_params"] = getattr(mcp_server.context, "params", {}).copy()

                # 智能调用：检查 func 是否接受 'ctx'
                sig = inspect.signature(func)
                if 'ctx' in sig.parameters:
                    result = await func(ctx, *args, **kwargs)
                else:
                    result = await func(*args, **kwargs)

                # 保存历史
                ctx.memory[history_key].append({"input": input_data, "result": result})
                return result

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(ctx, *args, **kwargs):
                if not hasattr(ctx, "memory"):
                    ctx.memory = {}
                ctx.memory.setdefault(history_key, [])

                input_data = extract_input(*((ctx,) + args), **kwargs) if extract_input else {"args": args, "kwargs": kwargs}
                input_data["context_params"] = getattr(mcp_server.context, "params", {}).copy()

                # 修正：不应将 ctx 传递给原始函数
                result = func(*args, **kwargs)
                ctx.memory[history_key].append({"input": input_data, "result": result})
                return result

            return sync_wrapper

    return decorator

# ----------------- MCP 历史记录提取函数 -----------------
def extract_kriging_input(ctx, points, **kwargs):
    """
    MCP 历史记录提取函数，兼容所有关键字参数
    """
    return {
        "points_count": len(points),
        "method": kwargs.get("method", "ok"),
        "candidate_models": kwargs.get("candidate_models", ["spherical", "exponential", "gaussian"]),
        "autoOptimizeModel": kwargs.get("autoOptimizeModel", False),
        "drift": kwargs.get("drift", "linear")
    }

# ---------------------------
# NLP 解析工具
# ---------------------------
@mcp_server.tool()
@mcp_tool_history("nlp_history", extract_input=lambda ctx, user_text, context=None: {"user_text": user_text})
def parse_text_tool_mcp(user_text: str, context: dict | None = None) -> Dict[str, Any]:
    """
    NLP 解析工具，支持黑板式上下文
    """
    result = parse_text_tool(user_text)
    if "task" in result and "warnings" not in result["task"]:
        result["task"]["warnings"] = []
    if "plan" in result and "errors" not in result["plan"]:
        result["plan"]["errors"] = []
    if context is not None:
        # 直接写入黑板上下文
        context["parsed_result"] = result
        context["task"] = result.get("task", {})
        context["plan"] = result.get("plan", {"pipeline": []})
    return result

# ---------------------------
# Text2SQL 工具
# ---------------------------

@mcp_tool_history("query_history", extract_input=lambda ctx, query: {"query": query})
@mcp_server.tool()
def text_to_sql_query_tool(ctx, query: str):
    return text_to_sql_query(query)


# ---------------------------
# Kriging 插值工具
# ---------------------------
interpolator = Interpolator()

@mcp_server.tool()
@mcp_tool_history(
    "kriging_history",
    extract_input=lambda ctx, points, **kwargs: {
        "points_count": len(points),
        "method": kwargs.get("method", "auto"),
        "candidate_models": kwargs.get("candidate_models", ["spherical", "exponential", "gaussian"]),
        "autoOptimizeModel": kwargs.get("autoOptimizeModel", True),
        "drift": kwargs.get("drift", "linear"),
        "idw_power": kwargs.get("idw_power", 2.0)
    }
)
async def kriging_interpolate(
    ctx,
    points: list[dict],
    method: str = "auto",
    candidate_models=None,
    autoOptimizeModel=True,
    drift="linear",
    idw_power: float = 2.0,
    **kwargs
):
    # === 🧩 调试语句：输出 points 的示例结构 ===
    print("\n[DEBUG] ====== Kriging 输入点信息 ======")
    try:
        # 尝试取前 3 个样本
        sample = points[:3] if isinstance(points, (list, tuple)) else list(points)[:3]
        print(f"[DEBUG] 输入点总数: {len(points)}")
        for i, p in enumerate(sample):
            print(f"[DEBUG] 示例点 {i}: 类型={type(p)} | 内容={repr(p)}")
            try:
                print(f"[DEBUG]    长度={len(p)}")
                if hasattr(p, "keys"):
                    print(f"[DEBUG]    键集合={list(p.keys())}")
            except Exception as e:
                print(f"[DEBUG]    长度检测异常: {e}")
    except Exception as e:
        print(f"[DEBUG] ❌ 无法打印输入点示例: {e}")
    print("[DEBUG] ==================================\n")

    if not points:
        return {"error": "缺少 points 数据"}

        # ------------------- ✅ 格式统一与预处理 -------------------
    parsed_points = []
    for p in points:
        if isinstance(p, dict):
            lon = p.get("lon") or p.get("x") or p.get("lng") or p.get("geo_X")
            lat = p.get("lat") or p.get("y") or p.get("geo_Y")
            val = p.get("value") or p.get("z") or p.get("v") or p.get("thickness")
            if lon is not None and lat is not None and val is not None:
                parsed_points.append({"lon": float(lon), "lat": float(lat), "value": float(val)})
        elif isinstance(p, (list, tuple)):
            if len(p) >= 3:
                parsed_points.append({"lon": float(p[0]), "lat": float(p[1]), "value": float(p[2])})
    if not parsed_points:
        return {"error": "坐标/数值提取失败: 输入点结构不符合要求"}

    points = parsed_points

    candidate_models = candidate_models or ["spherical", "exponential", "gaussian"]

    # ------------------- 提取坐标与值 -------------------
    try:
        lons = np.array([p.get("lon") or p.get("geo_X") for p in points], dtype=float)
        lats = np.array([p.get("lat") or p.get("geo_Y") for p in points], dtype=float)
        values = np.array([p.get("value") or p.get("thickness") for p in points], dtype=float)
        print(f"[INFO] 数据点数量: {len(values)}")
    except Exception as e:
        return {"error": f"坐标/数值提取失败: {e}"}

    # ------------------- 数据有效性 -------------------
    mask = np.isfinite(values)
    lons, lats, values = lons[mask], lats[mask], values[mask]
    if len(values) < 5:
        return {"error": "有效数据点不足，无法执行插值"}

    # ------------------- 自动判断插值方法 -------------------
    if method == "auto":
        suggestion = interpolator.suggest_kriging_method(lons, lats, values)
        method = suggestion["suggestion"]
        print(f"[INFO] 自动选择插值方法: {method.upper()} ({suggestion['reason']})")
    else:
        print(f"[INFO] 用户指定插值方法: {method.upper()}")

    # ------------------- 正态性检测 + Box-Cox -------------------

    values_transformed, lmbda, was_transformed, shift, shapiro_p = interpolator.check_normality_and_transform(values)
    mean, std = values_transformed.mean(), values_transformed.std()
    values_std = (values_transformed - mean) / (std if std > 0 else 1.0)
    boxcox_info = {"was_transformed": was_transformed, "lambda": lmbda, "shift": shift, "shapiro_p": shapiro_p}

    # ------------------- 自动优化半变异模型 (仅 Kriging) -------------------
    final_model = None
    model_scores = {}
    if method != "idw":
        print(f"[INFO] 正在选择最优半变异函数模型...")
        best_model, all_models = interpolator.select_best_model(lons, lats, values_std, candidate_models=candidate_models)
        if best_model is None:
            return {"error": "半变异函数模型选择失败", "details": all_models}
        print(f"[INFO] 最优半变异函数模型: {best_model}")

        # ------------------- 并行交叉验证 -------------------
        async def cv_task(model_name):
            print(f"[INFO] 正在进行交叉验证...")
            params = {"method": method, "variogram_model": model_name, "autoOptimizeModel": autoOptimizeModel, "drift": drift}
            try:
                n_points = len(values_std)
                sample_size = min(n_points, 30)
                idx = np.random.choice(n_points, sample_size, replace=False)
                cv_res = await asyncio.to_thread(interpolator.cross_validate, lons[idx], lats[idx], values[idx], params)
                return model_name, cv_res
            except Exception as e:
                return model_name, {"error": str(e)}

        tasks = [cv_task(m) for m in candidate_models]
        cv_results_list = await asyncio.gather(*tasks)

        model_scores = {}
        best_score = float("inf")
        best_cv_model = best_model
        for model_name, cv_res in cv_results_list:
            model_scores[model_name] = cv_res
            if "KRMSE" in cv_res and cv_res["KRMSE"] < best_score:
                best_cv_model = model_name
                best_score = cv_res["KRMSE"]
        final_model = best_cv_model or best_model
        print(f"[INFO] 最终采用模型: {final_model}")
    else:
        # IDW 不需要模型，所以跳过以上步骤
        final_model = None
        model_scores = {}

    final_params = {"method": method, "variogram_model": final_model, "autoOptimizeModel": autoOptimizeModel, "drift": drift, "idw_power": idw_power}

    # ------------------- 安全插值 -------------------
    def safe_interpolate(lons, lats, values, params, **kwargs):
        try:
            res = interpolator.interpolate(lons, lats, values, params, **kwargs)
            print("[DEBUG] 插值函数返回类型:", type(res))
            if isinstance(res, (tuple, list)):
                print("[DEBUG] 返回元素数量:", len(res))
                for i, item in enumerate(res):
                    print(f"   [DEBUG] 第 {i} 项类型: {type(item)}")
            elif isinstance(res, dict):
                print("[DEBUG] 返回为字典，键集合:", list(res.keys()))
        except Exception as e:
            return {"error": f"插值执行失败: {e}", "z": None, "ss": None, "grid_x": None, "grid_y": None}

        z = ss = grid_x = grid_y = None

        if isinstance(res, (tuple, list)):
            n = len(res)
            if n >= 1: z = res[0]
            if n >= 2: ss = res[1]
            if n >= 3: grid_x = res[2]
            if n >= 4: grid_y = res[3]
        elif isinstance(res, dict):
            z = res.get("z") or res.get("zk")
            ss = res.get("ss") or res.get("variance")
            grid_x = res.get("grid_x") or res.get("grid_lon")
            grid_y = res.get("grid_y") or res.get("grid_lat")
        else:
            return {"error": f"未知返回类型: {type(res)}", "z": None, "ss": None, "grid_x": None, "grid_y": None}

        # 自动生成网格
        if grid_x is None or grid_y is None:
            try:
                nx = ny = 100
                if np.ptp(lons) == 0: lons[0] += 1e-6
                if np.ptp(lats) == 0: lats[0] += 1e-6
                grid_x, grid_y = np.meshgrid(np.linspace(np.min(lons), np.max(lons), nx),
                                             np.linspace(np.min(lats), np.max(lats), ny))
            except Exception:
                grid_x = grid_y = None

        return {"z": z, "ss": ss, "grid_x": grid_x, "grid_y": grid_y}

    final_interp = await asyncio.to_thread(safe_interpolate, lons, lats, values, final_params, **kwargs)

    # ------------------- 汇总输出 -------------------
    final_interp.update({
        "boxcox_info": boxcox_info,
        "selected_method": method,
        "best_model": final_model,
        "cv_results": model_scores,
        "points_count": len(values)
    })

    return final_interp


# ---------------------------
# Kriging 叠加分析工具
# ---------------------------
@mcp_server.tool()
@mcp_tool_history(
    "overlay_history",
    extract_input=lambda ctx, datasets, weights, **kwargs: {
        "dataset_names": list(datasets.keys()),
        "weights": weights,
        "common_params": kwargs
    }
)
async def kriging_overlay_tool(
    ctx,
    datasets: Dict[str, Dict[str, List[float]]],
    weights: Dict[str, float],
    **kwargs
) -> Dict[str, Any]:
    """
    对多个数据集进行插值并进行加权叠加。
    `datasets`: {'泥岩': {'lons': [...], 'lats': [...], 'values': [...]}, ...}
    `weights`: {'泥岩': 0.6, '灰岩': 0.4}
    `kwargs`: 共享的插值参数
    """
    if not datasets or not weights:
        return {"error": "Datasets and weights are required."}

    try:
        # 注意：Interpolator() 是同步代码，但在异步函数中调用其同步方法是安全的
        # 如果 interpolate_and_overlay 是CPU密集型操作，未来可以考虑用 asyncio.to_thread
        result = interpolator.interpolate_and_overlay(
            datasets=datasets,
            weights=weights,
            common_params=kwargs
        )
        return result
    except Exception as e:
        return {"error": f"Overlay analysis failed: {str(e)}"}


# ---------------------------
# 地图渲染工具
# ---------------------------
renderer = MapRenderer()


@mcp_server.tool()
@mcp_tool_history(
    "map_render_history",
    extract_input=lambda ctx, grid_x, grid_y, z, points=None, **kwargs: {
        "grid_size": (len(grid_x), len(grid_y)),
        "points_count": len(points) if points else 0,
        "variable": kwargs.get("variable", "thickness"),
        "colormap": kwargs.get("colormap", "RdYlBu"),
        "n_classes": kwargs.get("n_classes", None)
    }
)
async def render_map_tool(ctx, grid_x, grid_y, z, first_result=None,
                          points=None, boundary_geom=None,
                          task_text=None, variable="thickness", lithology=None,
                          smooth_sigma=0, n_classes=11, colormap="RdYlBu",
                          lighten=False):
    result = await renderer.render_map(
        grid_x, grid_y, z, first_result,
        points, boundary_geom,
        task_text, variable, lithology,
        smooth_sigma, n_classes,
        colormap, lighten
    )

    return result


# ---------------------------
# 用户反馈解析工具
# ---------------------------
@mcp_server.tool()
@mcp_tool_history("feedback_history", extract_input=lambda ctx, feedback_text, context=None: {"feedback_text": feedback_text})
async def parse_user_feedback_tool(feedback_text: str, context: Optional[dict] = None) -> dict:
    result = await parse_user_feedback(feedback_text, context)
    if context is not None:
        for k, v in result.get("mcp_context", {}).get("params", {}).items():
            context[k] = v
    return result

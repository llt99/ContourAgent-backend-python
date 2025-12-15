# """
# Agent-4 用户反馈解析模块
# 结构：Agent + MCP
# 功能：
# - 接收自然语言反馈
# - 转换为 MCP 标准上下文参数
# - 支持触发任务重执行
# """

import asyncio
import re
from typing import Dict, Any

class FeedbackMCP:
    async def parse_feedback(self, feedback_text: str) -> Dict[str, Any]:
        """
        采用混合策略解析用户反馈：
        1. 优先解析结构化指令（如 "方法=普通克里金"）。
        2. 对未匹配的参数，回退到关键词扫描（如 "使用普通克里金"）。
        """
        feedback = {}
        
        # 参数和值的映射表
        param_mapping = {
            "方法": "method", "模型": "variogram_model", "色带": "colormap",
            "颜色": "colormap", "分级": "n_classes", "等值线": "n_classes",
            "区域": "region", "范围": "region",
        }
        value_mapping = {
            "method": {"普通克里金": "ok", "普通": "ok", "泛克里金": "uk", "泛": "uk", "趋势": "uk"},
            "variogram_model": {"球状": "spherical", "球形": "spherical", "高斯": "gaussian", "指数": "exponential"},
            "colormap": {"红黄绿": "RdYlGn", "蓝紫": "PuBu", "灰度": "Greys", "浅色": "viridis", "深色": "plasma", "默认": "RdYlBu"},
            "region": {"盆地": "basin", "全图": "full"}
        }

        # --- 策略1: 解析结构化指令 (key=value) ---
        pattern = re.compile(r"(\w+)\s*[:=]\s*([\u4e00-\u9fa5\w]+)")
        matches = pattern.findall(feedback_text)
        
        # 记录已被结构化指令处理过的参数
        handled_params = set()

        for key, value in matches:
            param_name = param_mapping.get(key)
            if not param_name:
                continue
            
            standard_value = value_mapping.get(param_name, {}).get(value, value)
            
            if param_name == "n_classes":
                try:
                    feedback[param_name] = int(standard_value)
                    handled_params.add(param_name)
                except (ValueError, TypeError):
                    pass
            else:
                feedback[param_name] = standard_value
                handled_params.add(param_name)

        # --- 策略2: 回退到关键词扫描 ---
        # 遍历所有可能的参数值，反向查找它们属于哪个参数
        for param_name, values_map in value_mapping.items():
            if param_name in handled_params:
                continue # 如果已通过结构化指令处理，则跳过
            
            for user_term, internal_value in values_map.items():
                if user_term in feedback_text:
                    feedback[param_name] = internal_value
                    break # 找到一个匹配就跳出内层循环

        # 对 n_classes 进行特殊关键词处理
        if "n_classes" not in handled_params:
            match = re.search(r"分(\d+)级|(\d+)个等级", feedback_text)
            if match:
                num_str = match.group(1) or match.group(2)
                feedback["n_classes"] = int(num_str)

        # --- 通用指令 ---
        # 检查是否明确要求重试
        re_execute_requested = "重试" in feedback_text or "重新" in feedback_text

        # 如果有任何参数被成功解析，也意味着需要重执行
        if len(feedback) > 0 or re_execute_requested:
            feedback["re_execute"] = True

        await asyncio.sleep(0)
        return feedback


feedback_parser = FeedbackMCP()

async def parse_user_feedback(feedback_text: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    parsed_params = await feedback_parser.parse_feedback(feedback_text)
    if context is None:
        context = {}
    context.setdefault("feedbackParsed", {})
    context["feedbackParsed"].update(parsed_params)
    context["last_feedback_agent"] = "FeedbackAgent"
    return {"mcp_context": {"agent_id": "FeedbackAgent", "task": "解析用户反馈", "params": parsed_params},
            "updated_context": context}


# ================= 运行入口 =================
# if __name__ == "__main__":
#     app.run()

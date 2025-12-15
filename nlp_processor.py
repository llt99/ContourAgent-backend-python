from __future__ import annotations
import json
from typing import Dict, Any
from openai import OpenAI
import httpx
import json5

# ------------------------------
# DeepSeek API 配置
# ------------------------------
transport = httpx.HTTPTransport(proxy="http://127.0.0.1:7897")

client = OpenAI(
    api_key="sk-25127b70dddd42ce8abe5c7faf7ae50e",
    base_url="https://api.deepseek.com/v1",
    http_client=httpx.Client(
        transport=transport,
        timeout=30.0
    )
)

# ------------------------------
# 系统 Prompt
# ------------------------------
SYSTEM_PROMPT = """
    你是地学绘图任务解析器。
    用户会给出一句中文绘图/分析需求。
    请严格返回 JSON（只返回 JSON，不带解释文字）。
    JSON 必须包含两个顶层字段："task" 和 "plan"。

    ## task 字段：
    必须输出以下键：
    - analysis_type: 分析类型。如果用户提到多个岩性（如“泥岩和灰岩”）、“叠加”、“综合评价”等，则为 "multi_factor"，否则为 "single_factor"。
    - region: 地理范围（如“四川盆地”）
    - stratum: 地层或组名（如“龙潭”）
    - variable: 主要的岩性或指标。如果是多因素，则为第一个或最重要的一个。
    - variables: **仅在 multi_factor 时出现**，是一个包含所有岩性/指标的列表。
    - plot: 图件类型（如“分布图”、“等值线图”）
    - method: 插值方法，如果用户未指定，请填 null
    - model: 半变异函数模型，如果用户未指定，请填 null
    - method_code: 插值方法代码，如果未指定填 null
    - model_code: 变异函数模型代码，如果未指定填 null
    - warnings: 解析中的提示信息（列表）
    - errors: 解析中的错误信息（列表）

    ## 实体代码映射规则：
    - **method_code**:
      - "普通克里金" or "OK" -> "ok"
      - "泛克里金" or "UK" -> "uk"
    - **model_code**:
      - "球状" or "spherical" -> "spherical"
      - "指数" or "exponential" -> "exponential"
      - "高斯" or "gaussian" -> "gaussian"
      - "线性" or "linear" -> "linear"

    # plan 字段：
    - pipeline: 一个字符串列表，包含按顺序需要执行的 Agent 名称。
    - 可用的 Agent 包括: "nlp", "data", "kriging", "overlay", "image", "feedback"。
    - **关键规则**: 如果 task.analysis_type 是 "multi_factor"，pipeline 必须包含 "overlay"。
    - 标准单因素绘图: ["nlp", "data", "kriging", "image"]
    - 标准多因素绘图: ["nlp", "data", "kriging", "overlay", "image"]

    - 单因素 JSON 示例：
    {
      "task": {
        "analysis_type": "single_factor",
        "region": "四川盆地",
        "stratum": "龙潭",
        "variable": "煤岩",
        "plot": "分布图",
        "method": null, "model": null, "method_code": null, "model_code": null,
        "warnings": [], "errors": []
      },
      "plan": { "pipeline": ["nlp", "data", "kriging", "image"] }
    }

    - 多因素 JSON 示例 (用户输入: "绘制四川盆地龙潭组泥岩和灰岩的综合评价图"):
    {
      "task": {
        "analysis_type": "multi_factor",
        "region": "四川盆地",
        "stratum": "龙潭",
        "variable": "泥岩",
        "variables": ["泥岩", "灰岩"],
        "plot": "综合评价图",
        "method": null, "model": null, "method_code": null, "model_code": null,
        "warnings": [], "errors": []
      },
      "plan": { "pipeline": ["nlp", "data", "kriging", "overlay", "image"] }
    }
"""


# ------------------------------
# JSON 提取辅助
# ------------------------------
def _find_json_block(text: str) -> str:
    """从文本中提取第一个 JSON 块"""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        if depth == 0:
            return text[start:i+1]
    return None

# ------------------------------
# 消息构造：是否加 system prompt
# ------------------------------
# def build_messages(user_text: str, is_initial: bool = True):
#     if is_initial:
#         return [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": user_text}
#         ]
#     else:
#         return [
#             {"role": "user", "content": user_text}
#         ]
#

# ------------------------------
# NLP 解析函数
# ------------------------------
def parse_text_tool(user_text: str, is_initial: bool = True) -> Dict[str, Any]:
    """
    核心 NLP 解析逻辑：
    - user_text: 用户输入文本
    - 返回 JSON dict，包含 task 和 plan
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0,
            max_tokens=800,
            timeout=600.0
        )
        content = resp.choices[0].message.content
        if not content:
            raise ValueError("DeepSeek 没有返回内容")

        # 尝试直接解析 JSON
        try:
            return json.loads(content)
        except:
            # 提取第一个 JSON 块
            jblock = _find_json_block(content)
            if not jblock:
                raise ValueError("LLM未返回 JSON 或 JSON 块无法提取:\n" + content)
            return json.loads(jblock)

    except Exception as e:
        return {
            "task": {
                "errors": [f"LLM解析失败: {str(e)}"]
            },
            "plan": {"status": "invalid", "errors": [str(e)], "pipeline": []}
        }

# ------------------------------
# 测试运行
# ------------------------------
if __name__ == "__main__":
    # 初始任务
    result1 = parse_text_tool("绘制四川盆地龙潭组煤岩分布图", is_initial=True)
    print("初始任务：")
    print(json.dumps(result1, ensure_ascii=False, indent=2))

    # 修改任务
    result2 = parse_text_tool("使用泛克里金方法重新绘图", is_initial=False)
    print("\n修改任务：")
    print(json.dumps(result2, ensure_ascii=False, indent=2))

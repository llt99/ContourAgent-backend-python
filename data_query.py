import logging
import re
from typing import Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from prompt import prompts

# LangChain 导入（新版）
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents.agent import AgentOutputParser

# 日志配置
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 数据库连接配置
DB_USER = "fastapi"
DB_PASS = "gisgisgis"
DB_HOST = "10.242.171.158"
DB_PORT = 3306
DB_NAME = "gdl"

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
    future=True
)

# 初始化 LangChain Text2SQL Agent
db = SQLDatabase.from_uri(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    openai_api_key="sk-25127b70dddd42ce8abe5c7faf7ae50e",
    openai_api_base="https://api.deepseek.com"
)

class SafeSQLParser(AgentOutputParser):
    def parse(self, text: str):
        # 强制只提取 SQL
        match = re.search(r"(SELECT .*?;)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return {"output": match.group(1)}
        return {"output": text}

text_to_sql_agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    handle_parsing_errors=True
)


# SQL 提取工具

def extract_sql(sql_text: str) -> str:
    """
    从 LLM 输出中提取 SQL 语句，自动处理代码块、解释、多余换行等情况。
    """
    logger.debug(f"原始 LLM 输出: {sql_text}")

    # 去掉 Markdown 代码块
    if "```" in sql_text:
        sql_text = re.sub(r"```[a-zA-Z]*", "", sql_text)
        sql_text = sql_text.replace("```", "").strip()

    # 匹配标准 SQL（带分号）
    match = re.search(
        r"(SELECT[\s\S]+?;|INSERT[\s\S]+?;|UPDATE[\s\S]+?;|DELETE[\s\S]+?;)",
        sql_text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    # 容错：无分号，取到结尾
    match = re.search(
        r"(SELECT[\s\S]+|INSERT[\s\S]+|UPDATE[\s\S]+|DELETE[\s\S]+)",
        sql_text,
        re.IGNORECASE
    )
    if match:
        return match.group(1).strip()

    raise RuntimeError(f"未能提取 SQL: {sql_text}")



# -----------------Text2SQL 查询工具-------------------
def text_to_sql_query(query: str) -> Dict[str, Any]:
    try:
        # 识别查询中的地层和岩性
        stratum_match = re.search(r"(龙潭|长兴|飞一|飞二|飞三|飞四)", query)
        stratum = stratum_match.group(1).replace("组", "") if stratum_match else "龙潭" # 默认龙潭

        rock_types = re.findall(r"(玄武岩|碳酸盐岩|碎屑岩|煤岩|膏盐岩|硅岩)", query)
        unique_rock_types = sorted(list(set(rock_types)))

        # 如果识别出多种岩性，则使用模板生成统一查询SQL
        if len(unique_rock_types) > 1:
            logger.info(f"检测到多种岩性: {unique_rock_types}，使用统一查询模板。")

            thickness_cols = ",\n    ".join([
                f"COUNT(DISTINCT CASE WHEN wl.cType1 LIKE '%{rock}%' THEN wl.dDepth END) * 0.125 AS {rock}_thickness"
                for rock in unique_rock_types
            ])
            ratio_cols = ",\n    ".join([
                f"(COUNT(DISTINCT CASE WHEN wl.cType1 LIKE '%{rock}%' THEN wl.dDepth END) * 0.125) / NULLIF((s.end_depth - s.start_depth), 0) AS {rock}_ratio"
                for rock in unique_rock_types
            ])

            sql = f"""
                SELECT
                    wt.well_name,
                    wt.Lon AS lon,
                    wt.Lat AS lat,
                    lt.stratum AS stratum,
    
                    -- ★ 动态岩性厚度
                    {thickness_cols},
    
                    -- ★ 地层总厚度
                    SUM(lt.cThick) AS total_thickness,
    
                    -- ★ 动态岩性占比
                    {ratio_cols}
    
                FROM well_test wt
                JOIN lith_test lt ON lt.well_name = wt.well_name
    
                WHERE lt.stratum LIKE '%{stratum}%'
    
                GROUP BY wt.well_name, wt.Lon, wt.Lat, lt.stratum
                ORDER BY wt.well_name, lt.stratum;
                """
            sql = sql.strip()
            logger.info(f"✅ 生成的统一查询 SQL:\n{sql}")

        # 如果只有一种或没有岩性，则使用原有的LLM查询逻辑
        else:
            logger.info("检测到单一岩性或无岩性，使用 LLM 进行查询。")
            examples_text = ""
            for item in prompts:
                examples_text += f"示例:\n用户输入: {item['user_input']}\nSQL: {item['sql']}\n\n"

            if len(unique_rock_types) == 1:
                thickness_instruction = """- 如果涉及岩性占比或厚度，请在 SELECT 子句中计算 thickness，例如：
                    - 厚度类: COUNT(DISTINCT wl.dDepth) * 0.125 AS thickness
                    - 地层厚度: (s.end_depth - s.start_depth) AS thickness"""
            else: # 比如只查地层厚度
                thickness_instruction = "- thickness 指的是地层厚度: (s.end_depth - s.start_depth)"

            input_prompt = f"""
                你是专业的 MySQL 助手。请根据用户自然语言问题生成对应 SQL。
                以下是训练示例（参考）：
                {examples_text}
                
                用户问题: {query}
                要求：
                - 只能使用数据库实际存在的表和字段
                - SQL 必须以分号结尾
                - 不要输出解释
                - 如果涉及地层，请在 WHERE 子句使用类似 '%{stratum}%' 的匹配
                - 输出完整 SQL
                {thickness_instruction}
                """

            llm_response = llm.invoke(input_prompt)
            sql_text = llm_response.content.strip()
            logger.info(f"LLM 原始输出: {sql_text}")
            sql = extract_sql(sql_text)
            sql = re.sub(r"\w+_thickness", "thickness", sql, flags=re.IGNORECASE)
            sql = re.sub(r"\w+_ratio", "ratio", sql, flags=re.IGNORECASE)
            logger.info(f"✅ 修正后的 SQL: {sql}")

        # ---------- 执行 SQL ----------
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()

        logger.info(f"[TEST] 查询结果条数: {len(rows)} 条")

        return {
            "sql": sql,
            "rows": [dict(r) for r in rows],
            "rows_count": len(rows)
        }


    except SQLAlchemyError as e:
        logger.error(f"MySQL 执行失败: {e}")
        raise RuntimeError(f"MySQL 执行失败: {e}")
    except Exception as e:
        logger.error(f"Text_to_SQL 查询失败: {e}")
        raise RuntimeError(f"Text_to_SQL 查询失败: {e}")

# ------------------------------
# 启动 MCP 服务端并运行测试
# ------------------------------
if __name__ == "__main__":
    query = "绘制四川盆地龙潭组煤岩分布图"
    try:
        result = text_to_sql_query(query)
        print("测试结果:")
        print("生成 SQL:", result["sql"])
        print("返回行数:", len(result["rows"]))
        if result["rows"]:
            print("前 3 条数据:", result["rows"][:3])
    except Exception as e:
        print("工具调用失败:", e)

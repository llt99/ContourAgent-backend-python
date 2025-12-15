# # prompt.py
# # 用于生成智能地质制图任务训练样例（支持系统层级合并）
# # 当输入为“绘制川东二叠系灰岩分布图”时，自动合并龙潭组 + 长兴组进行查询
#
# import random
# import textwrap
#
# prompts = []
#
# # ------------------------------
# # 研究区、地层体系、变量配置
# # ------------------------------
# REGIONS = ["四川盆地", "川东", "川西", "川南", "川北", "川中"]
#
# SYSTEM_TO_STRATA = {
#     "二叠系": ["龙潭组", "长兴组"],
#     "三叠系": ["飞一", "飞二", "飞三", "飞四"]
# }
#
# STRATA = ["龙潭组", "长兴组", "飞一", "飞二", "飞三", "飞四"]
#
# VARIABLES = [
#     "地层厚度", "煤岩厚度", "玄武岩厚度", "碳酸盐岩厚度", "碎屑岩厚度", "膏盐岩厚度",
#     "煤岩占比", "玄武岩占比", "碳酸盐岩占比", "碎屑岩占比", "膏盐岩占比",
#     "煤岩分布", "玄武岩分布", "碳酸盐岩分布", "碎屑岩分布", "膏盐岩分布"
#     "GR曲线", "AC曲线"
# ]
#
# LITHOLOGY_MAP = {
#     "coal": "煤岩",
#     "bas": "玄武岩",
#     "carb": "碳酸盐岩",
#     "cla": "碎屑岩",
#     "gyp": "膏盐岩",
#     "sil": "硅岩",
# }
#
# TEMPLATES = [
#     "绘制{region}{target}{var}分布图",
#     "展示{region}{target}{var}等值线",
#     "生成{region}{target}{var}地图",
#     "查询{region}{target}的{var}情况",
#     "绘制{target}{var}等值线图"
# ]
#
#
# # ------------------------------
# # 核心逻辑
# # ------------------------------
# count = 0
# for region in REGIONS:
#     for system, strata_list in SYSTEM_TO_STRATA.items():
#         for var in VARIABLES:
#             if count >= 200:
#                 break
#
#             # 50% 概率使用系统层级（如“二叠系”），50% 使用单组（如“龙潭组”）
#             if random.random() < 0.5:
#                 target = system
#                 strata = strata_list  # 合并层级
#             else:
#                 target = random.choice(strata_list)
#                 strata = [target]
#
#             # ------------------------------
#             # 构造自然语言输入
#             # ------------------------------
#             user_input = random.choice(TEMPLATES).format(region=region, target=target, var=var)
#
#             # ------------------------------
#             # NLP 解析结果
#             # ------------------------------
#             nlp_result = {
#                 "region": region,
#                 "system": system if target == system else None,
#                 "strata": strata,
#                 "variable": var
#             }
#
#             # ------------------------------
#             # SQL 构建逻辑
#             # ------------------------------
#             # 拼接地层 WHERE 子句（支持系统合并）
#             stratum_conditions = " OR ".join(
#                 [f"s.name LIKE '%{s.replace('组', '')}%'" for s in strata]
#             )
#
#             sql = "-- 未定义变量类型"
#
#             # 1️⃣ 地层厚度
#             if var == "地层厚度":
#                 sql = f"""
#                     SELECT
#                         w.well_name,
#                         s.name AS stratum_name,
#                         (s.end_depth - s.start_depth) AS thickness,
#                         w.geo_X AS lon,
#                         w.geo_Y AS lat
#                     FROM well w
#                     JOIN stratum s ON w.well_no = s.well_no
#                     WHERE ({stratum_conditions})
#                       AND (s.end_depth - s.start_depth) IS NOT NULL
#                       AND w.region LIKE '%{region}%';
#                 """
#
#             # 2️⃣ 岩性厚度
#             # elif var.endswith("厚度") and var != "地层厚度":
#             #     lith = var.replace("厚度", "")
#             #     sql = f"""
#             #         SELECT
#             #             w.well_name,
#             #             w.geo_X AS lon,
#             #             w.geo_Y AS lat,
#             #             s.name AS stratum_name,
#             #             COUNT(DISTINCT wl.dDepth) * 0.125 AS thickness
#             #         FROM well w
#             #         JOIN stratum s ON w.well_no = s.well_no
#             #         JOIN well_log wl ON wl.well_name = w.well_name
#             #         WHERE ({stratum_conditions})
#             #           AND wl.cType1 LIKE '%{lith}%'
#             #           AND wl.dDepth BETWEEN s.start_depth AND s.end_depth
#             #           AND w.region LIKE '%{region}%'
#             #         GROUP BY w.well_name, w.geo_X, w.geo_Y, s.name;
#             #     """
#
#             # # 3️⃣ 岩性分布
#             elif var.endswith("分布"):
#                 lith = var.replace("分布", "")
#                 sql = f"""
#                         SELECT
#                             wt.well_name,
#                             wt.Lon AS lon,
#                             wt.Lat AS lat,
#                             lt.stratum AS stratum_name,
#                             lt.cType1 AS lith_type,  -- 岩性类型
#                             SUM(lt.cThick) AS total_thickness  -- 煤岩总厚度
#                         FROM well_test wt
#                         JOIN lith_test lt ON lt.well_name = wt.well_name
#                         WHERE {stratum_conditions}  -- 根据传入的地层条件
#                           AND lt.cType1 LIKE '%{lith}%'  -- 根据传入的岩性类型
#                         GROUP BY wt.well_name, wt.Lon, wt.Lat
#                         ORDER BY wt.well_name;
#                     """
#
#             # 4️⃣ 占比（灰岩占比、泥岩占比）
#             # elif var.endswith("占比"):
#             #     lith = var.replace("占比", "")
#             #     sql = f"""
#             #         SELECT
#             #             w.well_name,
#             #             w.geo_X AS lon,
#             #             w.geo_Y AS lat,
#             #             s.name AS stratum_name,
#             #             (s.end_depth - s.start_depth) AS stratum_thickness,
#             #             COUNT(DISTINCT wl.dDepth) * 0.125 AS lith_thickness,
#             #             (COUNT(DISTINCT wl.dDepth) * 0.125) / NULLIF((s.end_depth - s.start_depth), 0) AS ratio
#             #         FROM well w
#             #         JOIN stratum s ON w.well_no = s.well_no
#             #         JOIN well_log wl ON wl.well_name = w.well_name
#             #         WHERE ({stratum_conditions})
#             #           AND wl.cType1 LIKE '%{lith}%'
#             #           AND wl.dDepth BETWEEN s.start_depth AND s.end_depth
#             #           AND w.region LIKE '%{region}%'
#             #         GROUP BY w.well_name, w.geo_X, w.geo_Y, s.name, s.start_depth, s.end_depth
#             #         HAVING ratio > 0;
#             #     """
#
#             # 5️⃣ 曲线类（GR, AC）
#             elif var in ["GR曲线", "AC曲线"]:
#                 col = var.replace("曲线", "")
#                 sql = f"""
#                     SELECT wl.dDepth, wl.{col}, s.name AS stratum_name
#                     FROM well_log wl
#                     JOIN stratum s ON wl.well_no = s.well_no
#                     WHERE ({stratum_conditions})
#                       AND wl.dDepth BETWEEN s.start_depth AND s.end_depth
#                     ORDER BY wl.dDepth;
#                 """
#
#             prompts.append({
#                 "user_input": user_input.strip(),
#                 "nlp_result": nlp_result,
#                 "sql": textwrap.dedent(sql).strip()
#             })
#
#             count += 1
#
#         if count >= 300:
#             break
#     if count >= 300:
#         break
#
# print(f"✅ 已生成 {len(prompts)} 条训练示例")
#
# # ------------------------------
# # 示例输出
# # ------------------------------
# if __name__ == "__main__":
#     for i, p in enumerate(random.sample(prompts, k=5)):
#         print(f"\n[{i+1}] 用户输入：{p['user_input']}")
#         print("NLP解析结果：", p['nlp_result'])
#         print("SQL：\n", p['sql'])


import random
import textwrap

prompts = []

# ------------------------------
# 研究区、地层体系、变量配置
# ------------------------------
REGIONS = ["四川盆地", "川东", "川西", "川南", "川北", "川中"]

SYSTEM_TO_STRATA = {
    "二叠系": ["龙潭组", "长兴组"],
    "三叠系": ["飞一", "飞二", "飞三", "飞四"]
}

STRATA = ["龙潭组", "长兴组", "飞一", "飞二", "飞三", "飞四"]

VARIABLES = [
    "地层厚度", "煤岩厚度", "玄武岩厚度", "碳酸盐岩厚度", "碎屑岩厚度", "膏盐岩厚度",
    "煤岩占比", "玄武岩占比", "碳酸盐岩占比", "碎屑岩占比", "膏盐岩占比",
    "煤岩分布", "玄武岩分布", "碳酸盐岩分布", "碎屑岩分布", "膏盐岩分布",
    "GR曲线", "AC曲线"
]

LITHOLOGY_MAP = {
    "coal": "煤岩",
    "bas": "玄武岩",
    "carb": "碳酸盐岩",
    "cla": "碎屑岩",
    "gyp": "膏盐岩",
    "sil": "硅岩",
}

TEMPLATES = [
    "绘制{region}{target}{var}分布图",
    "展示{region}{target}{var}等值线",
    "生成{region}{target}{var}地图",
    "查询{region}{target}的{var}情况",
    "绘制{target}{var}等值线图"
]

# ------------------------------
# 核心逻辑
# ------------------------------
count = 0
for region in REGIONS:
    for system, strata_list in SYSTEM_TO_STRATA.items():
        for var in VARIABLES:
            if count >= 200:
                break

            # 50% 概率使用系统层级（如“二叠系”），50% 使用单组（如“龙潭组”）
            if random.random() < 0.5:
                target = system
                strata = strata_list  # 合并层级
            else:
                target = random.choice(strata_list)
                strata = [target]

            # ------------------------------
            # 构造自然语言输入
            # ------------------------------
            user_input = random.choice(TEMPLATES).format(region=region, target=target, var=var)

            # ------------------------------
            # NLP 解析结果
            # ------------------------------
            nlp_result = {
                "region": region,
                "system": system if target == system else None,
                "strata": strata,
                "variable": var
            }

            # ------------------------------
            # SQL 构建逻辑
            # ------------------------------
            # 拼接地层 WHERE 子句（支持系统合并）
            stratum_conditions = " OR ".join(
                [f"s.name LIKE '%{s.replace('组', '')}%'" for s in strata]
            )

            sql = "-- 未定义变量类型"

            # 1️⃣ 地层厚度
            if var == "地层厚度":
                sql = f"""
                    SELECT 
                        wt.well_name, 
                        st.stratum AS stratum_name, 
                        (st.stratum_Z1Depth - st.stratum_Z0Depth) AS thickness,
                        wt.Lon AS lon, 
                        wt.Lat AS lat
                    FROM wt 
                    JOIN st ON wt.well_ID = st.well_ID
                    WHERE ({stratum_conditions})
                      AND (st.stratum_Z1Depth - st.stratum_Z0Depth) IS NOT NULL
                """

            # 2️⃣ 岩性分布
            elif var.endswith("分布"):
                lith = var.replace("分布", "")
                sql = f"""
                        SELECT
                            wt.well_name,
                            wt.Lon AS lon,
                            wt.Lat AS lat,
                            lt.stratum AS stratum_name,
                            lt.cType1 AS lith_type,  -- 岩性类型
                            SUM(lt.cThick) AS total_thickness  -- 岩性总厚度
                        FROM well_test wt
                        JOIN lith_test lt ON lt.well_name = wt.well_name
                        WHERE {stratum_conditions}  -- 根据传入的地层条件
                          AND lt.cType1 LIKE '%{lith}%'  -- 根据传入的岩性类型
                        GROUP BY wt.well_name, wt.Lon, wt.Lat
                        ORDER BY wt.well_name;
                    """

            prompts.append({
                "user_input": user_input.strip(),
                "nlp_result": nlp_result,
                "sql": textwrap.dedent(sql).strip()
            })

            count += 1

        if count >= 300:
            break
    if count >= 300:
        break

print(f"✅ 已生成 {len(prompts)} 条训练示例")

# ------------------------------
# 示例输出
# ------------------------------
if __name__ == "__main__":
    for i, p in enumerate(random.sample(prompts, k=5)):
        print(f"\n[{i+1}] 用户输入：{p['user_input']}")
        print("NLP解析结果：", p['nlp_result'])
        print("SQL：\n", p['sql'])

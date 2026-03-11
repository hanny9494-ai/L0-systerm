#!/usr/bin/env python3
"""
修正 stage2_embedding.py 中的 DOMAIN_PREFERRED_CHAPTERS 映射
基于实际章节内容分析重新配置
"""

import re

TARGET = "/Users/jeff/l0-knowledge-engine/stage2_embedding.py"

NEW_MAPPING = '''DOMAIN_PREFERRED_CHAPTERS = {
    "heat_transfer":     [13, 25],       # ch13=烹饪热传递/褐变, ch25=传热方式/风味
    "chemical_reaction": [25, 21, 19],   # ch25=褐变反应, ch21=糖化学, ch19=发酵化学
    "physical_change":   [13, 25, 26],   # ch13=热处理物理变化, ch25=烹饪, ch26=水化学
    "water_activity":    [26],           # ch26=水的化学性质
    "protein_science":   [7, 10, 11],    # ch7=乳蛋白, ch10=肉类蛋白, ch11=肉类加热
    "lipid_science":     [7, 20],        # ch7=乳脂肪, ch20=酱汁/乳化
    "carbohydrate":      [18, 19, 21],   # ch18=谷物豆类, ch19=面团淀粉, ch21=糖
    "enzyme":            [7, 9, 19],     # ch7=乳酶, ch9=鸡蛋, ch19=面包发酵酶
    "flavor_sensory":    [17, 25],       # ch17=香料风味化学, ch25=风味形成
    "fermentation":      [19, 23],       # ch19=面包发酵, ch23=酒精/醋酸发酵
    "food_safety":       [10, 11, 14],   # ch10=肉类安全, ch11=加热杀菌, ch14=果蔬储存
    "emulsion_colloid":  [20, 7],        # ch20=酱汁乳化/明胶, ch7=乳制品胶体
    "color_pigment":     [11, 14, 25],   # ch11=肉色, ch14=果蔬色素, ch25=褐变
    "equipment_physics": [13, 25],       # ch13=烹饪设备热传递, ch25=传热方式
}'''

# 读取文件
with open(TARGET, "r", encoding="utf-8") as f:
    content = f.read()

# 替换 DOMAIN_PREFERRED_CHAPTERS 块
pattern = r'DOMAIN_PREFERRED_CHAPTERS\s*=\s*\{[^}]+\}'
if not re.search(pattern, content, re.DOTALL):
    print("❌ 未找到 DOMAIN_PREFERRED_CHAPTERS，请检查文件")
    exit(1)

new_content = re.sub(pattern, NEW_MAPPING, content, flags=re.DOTALL)

# 同时排除噪声章节：在 cosine_similarity 计算后，跳过 ch15/ch28/ch29
NOISE_FILTER = '''
        # 跳过噪声章节（参考文献、版权页、纯列表）
        if meta["chapter"] in (15, 28, 29):
            continue
'''

# 在 scored = [] 后插入噪声过滤
new_content = new_content.replace(
    "        scored = []\n        for meta in chunk_meta:",
    "        scored = []\n        for meta in chunk_meta:" + NOISE_FILTER
)

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(new_content)

print("✅ 已更新 DOMAIN_PREFERRED_CHAPTERS")
print("✅ 已添加噪声章节过滤 (ch15, ch28, ch29)")
print()
print("新映射：")
for line in NEW_MAPPING.split("\n"):
    if ":" in line and "#" in line:
        print(" ", line.strip())
print()
print("运行：python3 /Users/jeff/l0-knowledge-engine/stage2_embedding.py")

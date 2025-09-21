import os
import pandas as pd

# 确保有 data 文件夹
os.makedirs('data', exist_ok=True)

# 示例数据（8 条）
samples = [
    ("这部电影太好看了，我很喜欢", 1),
    ("剧情很差，浪费时间", 0),
    ("演员演技在线，推荐观看", 1),
    ("太烂了，再也不会看第二遍", 0),
    ("画面漂亮，适合周末放松", 1),
    ("矛盾很多，结局莫名其妙", 0),
    ("配乐棒，故事感人", 1),
    ("节奏拖沓，台词尴尬", 0),
]

# 转成表格并保存为 CSV
df = pd.DataFrame(samples, columns=['text','label'])
df.to_csv('data/train.csv', index=False, encoding='utf-8')

print("已生成 data/train.csv（8 条示例数据）")

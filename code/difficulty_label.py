import os
import pandas as pd
import matplotlib.pyplot as plt

# 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# 检查是否存在特征文件
FEATURES_PATH = os.path.join(DATA_DIR, "features.csv")
if not os.path.exists(FEATURES_PATH):
    print("错误: 找不到 features.csv。请先运行 feature_engineering.py 生成特征。")
    exit(1)

df = pd.read_csv(FEATURES_PATH)

print("--- 难度标签分析 ---")
print(df[["Word", "avg_tries", "difficulty"]].describe())

print("\n最难的5个词：")
print(df.nlargest(5, "avg_tries")[["Word", "avg_tries"]])

print("\n最简单的5个词：")
print(df.nsmallest(5, "avg_tries")[["Word", "avg_tries"]])

# 难度分布验证
print("\n难度分布计数：")
# 0: Easy, 1: Medium, 2: Hard
difficulty_counts = df["difficulty"].value_counts().sort_index()
print(difficulty_counts)

# 画分布图
plt.figure(figsize=(10, 5))
plt.hist(df["avg_tries"], bins=30, color="teal", edgecolor="white", alpha=0.7)
plt.axvline(df[df["difficulty"]==0]["avg_tries"].max(), color='green', linestyle='--', label='Easy|Medium')
plt.axvline(df[df["difficulty"]==1]["avg_tries"].max(), color='red', linestyle='--', label='Medium|Hard')
plt.title("Distribution of Difficulty (avg_tries)")
plt.xlabel("Average Tries")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()

# 自动保存
plot_path = os.path.join(DATA_DIR, "difficulty_distribution.png")
plt.savefig(plot_path, dpi=150)
print(f"\n分布图已保存至: {plot_path}")
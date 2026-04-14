import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

def run_eda():
    df = pd.read_excel(os.path.join(DATA_DIR, "Problem_C_Data_Wordle.xlsx"), header=1)
    df = df.sort_values("Date").reset_index(drop=True)
    df["hard_mode_ratio"] = df["Number in hard mode"] / df["Number of  reported results"]

    # 确认异常点
    idx = df.loc[df["hard_mode_ratio"] > 0.5].index[0]
    print(df.iloc[idx-3 : idx+4][["Date", "Word", "Number of  reported results",
                                   "Number in hard mode", "hard_mode_ratio"]])

    # 清洗
    df["Word"] = df["Word"].str.strip().str.lower()
    
    # 修复数据集中的拼写错误
    df.loc[df["Word"] == "tash", "Word"] = "trash"
    df.loc[df["Word"] == "clen", "Word"] = "clean"

    outlier_date = pd.Timestamp("2022-11-30")
    df_clean = df[df["Date"] != outlier_date].reset_index(drop=True)
    print(f"清洗后数据量：{len(df_clean)} 行")
    
    # 再次检查字长
    word_lens = df_clean["Word"].str.len().value_counts()
    print("字长统计：")
    print(word_lens)

    # 保存
    df_clean.to_csv(os.path.join(DATA_DIR, "data_clean.csv"), index=False)
    print("已保存 data_clean.csv")

    # 画图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(df_clean["Date"], df_clean["Number of  reported results"])
    axes[0].set_title("Daily Reported Results")
    axes[1].plot(df_clean["Date"], df_clean["hard_mode_ratio"], color="orange")
    axes[1].set_title("Hard Mode Ratio Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "eda_clean.png"), dpi=150)
    print("图已保存")

# 关键：只有直接运行这个文件时才执行
if __name__ == "__main__":
    run_eda()
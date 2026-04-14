import os
import math
import pandas as pd
import nltk
from wordfreq import word_frequency

# 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

def run_feature_engineering():
    # 读清洗后的数据
    df = pd.read_csv(os.path.join(DATA_DIR, "data_clean.csv"))
    df["Date"] = pd.to_datetime(df["Date"])

    # --- 特征0：难度标签 (Difficulty Labeling) ---
    # 先算出每个词的平均尝试次数，再以此划分难度
    tries_cols = ["1 try", "2 tries", "3 tries", "4 tries", "5 tries", "6 tries", "7 or more tries (X)"]
    weights = [1, 2, 3, 4, 5, 6, 7]
    df["avg_tries"] = sum(df[c] * w for c, w in zip(tries_cols, weights)) / 100

    # 基于分位数划分难度（0-简单, 1-中等, 2-困难）
    q25 = df["avg_tries"].quantile(0.25)
    q75 = df["avg_tries"].quantile(0.75)

    def get_label(x):
        if x < q25: return 0
        elif x < q75: return 1
        else: return 2

    df["difficulty"] = df["avg_tries"].apply(get_label)
    print(f"难度划分完成：Easy(0) < {q25:.2f} <= Medium(1) < {q75:.2f} <= Hard(2)")
    print(df["difficulty"].value_counts().sort_index())

    # --- 特征1：词频 ---
    df["word_freq"] = df["Word"].apply(lambda w: word_frequency(w, "en"))

    # --- 特征2：重复字母数 ---
    def count_repeat_letters(word):
        word = word.lower()
        return len(word) - len(set(word))

    df["repeat_letters"] = df["Word"].apply(count_repeat_letters)

    # --- 特征3：信息熵 ---
    def calc_entropy(word):
        word = word.lower()
        freq = {}
        for c in word:
            freq[c] = freq.get(c, 0) + 1
        entropy = 0
        for count in freq.values():
            p = count / len(word)
            entropy -= p * math.log2(p)
        return entropy

    df["entropy"] = df["Word"].apply(calc_entropy)

    # 看一下结果
    print(df[["Word", "word_freq", "repeat_letters", "entropy"]].head(10))

    # --- 特征4：Hard Mode 比例（7天滚动中位数归一化）---
    df = df.sort_values("Date").reset_index(drop=True)
    df["hard_mode_ratio"] = df["Number in hard mode"] / df["Number of  reported results"]

    rolling_median = df["hard_mode_ratio"].rolling(window=7, min_periods=1).median()
    df["hard_mode_ratio_norm"] = df["hard_mode_ratio"] / rolling_median

    # --- 特征5：首字母候选空间 ---
    nltk.download("words", quiet=True)
    from nltk.corpus import words as nltk_words
    word_list = [w.lower() for w in nltk_words.words() if len(w) == 5]
    from collections import Counter
    first_letter_counts = Counter(w[0] for w in word_list)

    df["first_letter_space"] = df["Word"].apply(
    lambda w: first_letter_counts.get(w[0].lower(), 0)
)

    # --- 特征7：位置信息熵 ---

    # 用nltk词库统计每个位置的字母频率
    position_freq = [Counter() for _ in range(5)]
    for word in word_list:  # word_list 前面已经构造好了
        for i, c in enumerate(word):
            position_freq[i][c] += 1

    def positional_entropy(word):
        word = word.lower()
        if len(word) != 5:
            return 0  # 异常处理
        score = 0
        for i, c in enumerate(word):
            total = sum(position_freq[i].values())
            p = position_freq[i].get(c, 1) / total
            score -= math.log2(p)
        return score

    # 再次确保只处理 5 位单词（双重保险）
    df = df[df["Word"].str.len() == 5].reset_index(drop=True)
    df["positional_entropy"] = df["Word"].apply(positional_entropy)

    # 验证直觉
    print("\n位置信息熵验证：")
    for word in ["eerie", "train", "crane", "parer", "nymph"]:
        pe = positional_entropy(word)
        print(f"{word}: {pe:.4f}")

    # --- 特征6：与常见开局词的字母重叠数 ---
    opener_words = ["crane", "stare", "audio", "raise", "slate"]

    def letter_overlap(word, opener):
        """谜底与开局词的字母集合交集大小"""
        return len(set(word.lower()) & set(opener.lower()))

    # 优化计算逻辑
    overlap_cols = []
    for opener in opener_words:
        col_name = f"overlap_{opener}"
        df[col_name] = df["Word"].apply(lambda w: letter_overlap(w, opener))
        overlap_cols.append(col_name)

    df["avg_opener_overlap"] = df[overlap_cols].mean(axis=1)

    # --- 验证直觉：简单词应该重叠更多 ---
    print("\n按难度分组的平均开局词重叠数 (验证特征有效性)：")
    print(df.groupby("difficulty")["avg_opener_overlap"].mean())

    # --- 保存最终特征表 ---
    final_cols = [
    "Date", "Word", "difficulty", "avg_tries",
    "word_freq", "repeat_letters", "entropy", 
    "hard_mode_ratio_norm", "first_letter_space", "avg_opener_overlap"
    ] + overlap_cols + tries_cols

    df_final = df[final_cols]
    df_final.to_csv(os.path.join(DATA_DIR, "features.csv"), index=False)

    # --- 子模型：用语言学特征预测 hard_mode_ratio_norm ---
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    lang_features = ["word_freq", "repeat_letters", "entropy",
                     "first_letter_space", "avg_opener_overlap", "positional_entropy"]

    X_lang = df[lang_features].values
    y_hard = df["hard_mode_ratio_norm"].values

    sub_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(sub_model, X_lang, y_hard,
                        cv=5, scoring="neg_mean_absolute_error")
    print(f"\n子模型MAE: {-scores.mean():.4f} ± {scores.std():.4f}")

    # 训练完整子模型，生成预测值
    sub_model.fit(X_lang, y_hard)
    df["hard_mode_ratio_pred"] = sub_model.predict(X_lang)

    print("\n真实值 vs 预测值对比：")
    print(df[["Word", "hard_mode_ratio_norm", "hard_mode_ratio_pred"]].head(10))

    print(f"\n特征工程完成，最终特征表已保存至 data/features.csv，形状: {df_final.shape}")
    print(df_final.head(5))

if __name__ == "__main__":
    run_feature_engineering()
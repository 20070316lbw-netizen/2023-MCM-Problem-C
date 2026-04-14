import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import os

# 设置seaborn风格
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_data():
    """加载数据"""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    df_clean = pd.read_csv(os.path.join(data_path, "data_clean.csv"))
    df_features = pd.read_csv(os.path.join(data_path, "features.csv"))
    return df_clean, df_features


def save_figure(fig, filename):
    """保存图片到output/figures目录"""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "figures")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved: {filepath}")


def plot_feature_correlation(df_features):
    """1. 特征相关性热力图"""
    features = [
        "word_freq",
        "repeat_letters",
        "entropy",
        "hard_mode_ratio_norm",
        "first_letter_space",
        "avg_opener_overlap",
        "avg_tries",
    ]

    # 计算相关性矩阵
    corr_matrix = df_features[features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, pad=20)
    plt.tight_layout()

    save_figure(fig, "feature_correlation.png")
    plt.close(fig)


def plot_feature_scatter(df_features):
    """2. 各特征与avg_tries的散点图（2行3列）"""
    features = [
        "word_freq",
        "repeat_letters",
        "entropy",
        "hard_mode_ratio_norm",
        "first_letter_space",
        "avg_opener_overlap",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        ax.scatter(
            df_features[feature],
            df_features["avg_tries"],
            alpha=0.6,
            s=30,
            edgecolor="w",
            linewidth=0.5,
        )

        # 添加趋势线
        z = np.polyfit(df_features[feature], df_features["avg_tries"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(
            df_features[feature].min(), df_features[feature].max(), 100
        )
        ax.plot(x_range, p(x_range), "r-", linewidth=2, alpha=0.8)

        ax.set_xlabel(feature.replace("_", " ").title())
        ax.set_ylabel("Average Tries")
        ax.set_title(f"{feature.replace('_', ' ').title()} vs Avg Tries")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "feature_scatter.png")
    plt.close(fig)


def plot_difficulty_boxplot(df_features):
    """3. 三档难度的avg_tries箱线图"""
    # 创建难度标签
    difficulty_labels = {0: "Easy", 1: "Medium", 2: "Hard"}
    df_features["difficulty_label"] = df_features["difficulty"].map(difficulty_labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 箱线图
    sns.boxplot(
        x="difficulty_label",
        y="avg_tries",
        data=df_features,
        order=["Easy", "Medium", "Hard"],
        ax=ax,
    )

    # 添加散点显示数据分布
    sns.stripplot(
        x="difficulty_label",
        y="avg_tries",
        data=df_features,
        order=["Easy", "Medium", "Hard"],
        size=4,
        color="black",
        alpha=0.3,
        jitter=True,
        ax=ax,
    )

    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Average Tries")
    ax.set_title("Distribution of Average Tries by Difficulty Level")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "difficulty_boxplot.png")
    plt.close(fig)


def plot_confusion_matrix(df_features):
    """4. 混淆矩阵（使用LightGBM 5折交叉验证）"""
    # 特征和标签
    feature_cols = [
        "word_freq",
        "repeat_letters",
        "hard_mode_ratio_norm",
        "first_letter_space",
        "avg_opener_overlap",
    ]
    X = df_features[feature_cols].values
    y = df_features["difficulty"].values

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "verbose": -1,
    }

    oof_preds = np.zeros((len(df_features), 3))

    for train_idx, val_idx in skf.split(X, y):
        train_set = lgb.Dataset(X[train_idx], label=y[train_idx])
        val_set = lgb.Dataset(X[val_idx], label=y[val_idx])
        model = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(-1)],
        )
        oof_preds[val_idx] = model.predict(X[val_idx])

    y_pred = oof_preds.argmax(axis=1)

    # 计算混淆矩阵
    cm = confusion_matrix(y, y_pred)

    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Easy", "Medium", "Hard"],
        yticklabels=["Easy", "Medium", "Hard"],
        ax=ax,
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (5-Fold Cross Validation)")

    plt.tight_layout()
    save_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Easy", "Medium", "Hard"]))


def plot_distribution_comparison(df_clean):
    """5. 单词预测分布条形图"""
    # 从数据中选择三个有代表性的单词：简单、中等、困难各一个
    # 首先查看数据中实际存在的单词
    easy_words = df_clean[
        df_clean["Word"].str.upper().isin(["DRINK", "PANIC", "SOLAR"])
    ]
    medium_words = df_clean[
        df_clean["Word"].str.upper().isin(["CRANK", "TANGY", "ROBOT"])
    ]
    hard_words = df_clean[
        df_clean["Word"].str.upper().isin(["GORGE", "QUERY", "ABBEY"])
    ]

    # 选择每个难度的一个单词
    words = []
    if len(easy_words) > 0:
        words.append(easy_words.iloc[0]["Word"].upper())
    if len(medium_words) > 0:
        words.append(medium_words.iloc[0]["Word"].upper())
    if len(hard_words) > 0:
        words.append(hard_words.iloc[0]["Word"].upper())

    if len(words) < 3:
        # 如果找不到足够的单词，使用前三个不同的单词
        words = df_clean["Word"].head(3).str.upper().tolist()

    word_data = []

    for word in words:
        word_row = df_clean[df_clean["Word"].str.upper() == word]
        if len(word_row) > 0:
            # 获取预测分布
            tries_cols = [
                "1 try",
                "2 tries",
                "3 tries",
                "4 tries",
                "5 tries",
                "6 tries",
                "7 or more tries (X)",
            ]
            distribution = word_row[tries_cols].iloc[0].values
            word_data.append((word, distribution))

    if not word_data:
        print("Warning: Could not find words in data")
        return

    # 创建条形图
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(tries_cols))
    width = 0.25

    for i, (word, dist) in enumerate(word_data):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, dist, width, label=word)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Number of Tries")
    ax.set_ylabel("Percentage (%)")
    # 根据实际单词创建标题
    if len(word_data) == 3:
        title = f"Distribution Comparison: {word_data[0][0]} (Easy) vs {word_data[1][0]} (Medium) vs {word_data[2][0]} (Hard)"
    else:
        title = "Distribution Comparison of Selected Words"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "X"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, "distribution_comparison.png")
    plt.close(fig)


def plot_q2_mae():
    """6. 各问题二目标MAE对比图"""
    # 这里需要根据实际模型结果来设置MAE值
    # 暂时使用示例数据，实际使用时需要从模型输出中获取
    targets = [
        "1 try",
        "2 tries",
        "3 tries",
        "4 tries",
        "5 tries",
        "6 tries",
        "7 or more tries (X)",
    ]

    # 示例MAE值 - 实际应该从模型评估中获取
    mae_values = [0.85, 1.23, 1.56, 1.89, 2.12, 2.45, 2.78]

    fig, ax = plt.subplots(figsize=(10, 6))

    # 创建水平条形图
    y_pos = np.arange(len(targets))
    bars = ax.barh(y_pos, mae_values, color=sns.color_palette("husl", len(targets)))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(targets)
    ax.set_xlabel("Mean Absolute Error (MAE)")
    ax.set_title("MAE Comparison for Question 2 Targets")
    ax.invert_yaxis()  # 从上到下显示

    # 添加数值标签
    for i, (bar, mae) in enumerate(zip(bars, mae_values)):
        width = bar.get_width()
        ax.text(
            width + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{mae:.3f}",
            va="center",
            fontsize=10,
        )

    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_figure(fig, "q2_mae.png")
    plt.close(fig)


def run_visualization():
    """运行所有可视化"""
    print("Loading data...")
    df_clean, df_features = load_data()

    print("1. Plotting feature correlation heatmap...")
    plot_feature_correlation(df_features)

    print("2. Plotting feature scatter plots...")
    plot_feature_scatter(df_features)

    print("3. Plotting difficulty boxplot...")
    plot_difficulty_boxplot(df_features)

    print("4. Plotting confusion matrix...")
    plot_confusion_matrix(df_features)

    print("5. Plotting distribution comparison...")
    plot_distribution_comparison(df_clean)

    print("6. Plotting Q2 MAE comparison...")
    plot_q2_mae()

    print("\nAll visualizations completed!")


if __name__ == "__main__":
    run_visualization()

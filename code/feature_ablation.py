import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

def run_feature_ablation():
    df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    y = df["difficulty"].values

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "verbose": -1,
    }

    def run_cv(feature_cols):   
        X = df[feature_cols].values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros((len(df), 3))
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            train_set = lgb.Dataset(X[train_idx], label=y[train_idx])
        val_set = lgb.Dataset(X[val_idx], label=y[val_idx])
        model = lgb.train(
            params, train_set,
            num_boost_round=300,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(-1)],
        )
        oof_preds[val_idx] = model.predict(X[val_idx])
        return accuracy_score(y, oof_preds.argmax(axis=1))

    all_features = [
        "word_freq", "repeat_letters", "entropy",
        "hard_mode_ratio_norm", "first_letter_space",
        "avg_opener_overlap"
    ]

    results = []

    # 1. 每个特征单独跑
    print("=== 单特征 ===")
    for f in all_features:
        acc = run_cv([f])
        results.append({"features": f, "accuracy": acc})
        print(f"{f:30s}: {acc:.4f}")

    # 2. 组合测试
    print("\n=== 组合测试 ===")
    combos = {
        "原始全特征":                    ["word_freq","repeat_letters","entropy","hard_mode_ratio_norm","first_letter_space"],
        "加入overlap":                   ["word_freq","repeat_letters","hard_mode_ratio_norm","first_letter_space","avg_opener_overlap"],
        "去entropy+加overlap":           ["word_freq","repeat_letters","hard_mode_ratio_norm","first_letter_space","avg_opener_overlap"],
        "hard+freq+overlap":             ["hard_mode_ratio_norm","word_freq","avg_opener_overlap"],
        "hard+freq+overlap+repeat":      ["hard_mode_ratio_norm","word_freq","avg_opener_overlap","repeat_letters"],
        "全特征含overlap":                all_features,
    }

    for name, cols in combos.items():
        acc = run_cv(cols)
        results.append({"features": name, "accuracy": acc})
        print(f"{name:30s}: {acc:.4f}")

    # 汇总
    print("\n=== 汇总排名 ===")
    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    run_feature_ablation()
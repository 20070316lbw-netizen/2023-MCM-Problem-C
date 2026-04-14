import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

def run_model_q2():
    df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))

    # 特征和目标
    feature_cols = [
        "word_freq", "repeat_letters",
    "hard_mode_ratio_norm", "first_letter_space", "avg_opener_overlap"
]
    target_cols = ["1 try","2 tries","3 tries","4 tries","5 tries","6 tries","7 or more tries (X)"]

    X = df[feature_cols].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    params = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 5,
    "verbose": -1,
}

    # 对每个目标变量单独训练一个模型
    oof_preds = np.zeros((len(df), len(target_cols)))

    for t_idx, target in enumerate(target_cols):
        y = df[target].values
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            train_set = lgb.Dataset(X[train_idx], label=y[train_idx])
            val_set = lgb.Dataset(X[val_idx], label=y[val_idx])
            model = lgb.train(
                params, train_set,
                num_boost_round=300,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(-1)],
            )
            oof_preds[val_idx, t_idx] = model.predict(X[val_idx])

    # 评估每个目标的MAE
    print("各目标MAE：")
    for i, target in enumerate(target_cols):
        mae = mean_absolute_error(df[target].values, oof_preds[:, i])
        print(f"  {target:30s}: {mae:.3f}")

    # 预测 EERIE（手动构造特征）
    from wordfreq import word_frequency
    eerie_features = np.array([[
        word_frequency("eerie", "en"),  # word_freq
        2,                               # repeat_letters（3个E）
        1.2,                             # hard_mode_ratio_norm（假设略高于均值）
        254,                             # first_letter_space（E开头）
        1.4,                             # avg_opener_overlap
    ]])

    print("\nEERIE 预测分布：")
    eerie_pred = []
    for t_idx, target in enumerate(target_cols):
        # 用最后一折的模型预测
        pred = model.predict(eerie_features)[0]
        eerie_pred.append(pred)

        # 重新训练完整模型预测EERIE
        full_preds = []
        for t_idx, target in enumerate(target_cols):
            y = df[target].values
            train_set = lgb.Dataset(X, label=y)
            model_full = lgb.train(params, train_set, num_boost_round=100)
            pred = model_full.predict(eerie_features)[0]
            full_preds.append(max(0, pred))  # 百分比不能为负

        # 归一化到100%
        total = sum(full_preds)
        full_preds_norm = [p / total * 100 for p in full_preds]

        print(f"{'尝试次数':10s} {'预测%':>8s}")
        for target, pred in zip(target_cols, full_preds_norm):
            print(f"{target:30s}: {pred:.1f}%")
        print(f"合计: {sum(full_preds_norm):.1f}%")

if __name__ == "__main__":
    run_model_q2()
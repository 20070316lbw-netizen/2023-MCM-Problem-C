import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")


def run_model_q3():
    df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))

    # 特征和标签
    feature_cols = ["word_freq", "repeat_letters", "entropy", 
                    "hard_mode_ratio_norm", "first_letter_space"]
    X = df[feature_cols].values
    y = df["difficulty"].values

    # StratifiedKFold 保证每折三档比例一致
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "verbose": -1,
    }

    oof_preds = np.zeros((len(df), 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=300,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
        )
        oof_preds[val_idx] = model.predict(X_val)

    # 整体评估
    y_pred = oof_preds.argmax(axis=1)
    print("\n整体准确率：", accuracy_score(y, y_pred))
    print("\n分类报告：")
    print(classification_report(y, y_pred, target_names=["easy","medium","hard"]))

    # 特征重要性（用最后一折的模型）
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain")
    }).sort_values("importance", ascending=False)

    print("\n特征重要性：")
    print(importance)

    plt.figure(figsize=(8, 4))
    plt.barh(importance["feature"], importance["importance"])
    plt.title("Feature Importance (gain)")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "feature_importance.png"), dpi=150)
    print("\n图已保存")

if __name__ == "__main__":
    run_model_q3()
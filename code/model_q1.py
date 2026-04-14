import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output", "figures")


def create_features(df):
    """创建时序特征"""
    df = df.copy()

    # 确保按日期排序
    df = df.sort_values("Date").reset_index(drop=True)

    # 目标变量
    df["reported"] = df["Number of  reported results"]

    # 特征1: 星期几 (0=周一, 6=周日)
    df["day_of_week"] = df["Date"].dt.dayofweek

    # 特征2: 距数据起始天数
    df["days_from_start"] = (df["Date"] - df["Date"].min()).dt.days

    # 特征3: 距峰值天数
    peak_date = df.loc[df["reported"].idxmax(), "Date"]
    df["days_from_peak"] = (df["Date"] - peak_date).dt.days

    # 特征4: 前7天人数
    df["lag_7"] = df["reported"].shift(7)

    # 特征5: 前14天人数
    df["lag_14"] = df["reported"].shift(14)

    # 特征6: 过去7天滚动均值 (shift(1)避免数据泄露)
    df["rolling_mean_7"] = (
        df["reported"].rolling(window=7, min_periods=1).mean().shift(1)
    )

    # 特征7: 过去7天滚动标准差 (shift(1)避免数据泄露)
    df["rolling_std_7"] = df["reported"].rolling(window=7, min_periods=1).std().shift(1)

    # 添加月份特征
    df["month"] = df["Date"].dt.month

    # 添加是否周末特征
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def run_model_q1():
    """运行问题1时序预测模型"""

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读取数据
    df = pd.read_csv(os.path.join(DATA_DIR, "data_clean.csv"))
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"数据形状: {df.shape}")
    print(f"日期范围: {df['Date'].min()} 到 {df['Date'].max()}")

    # 创建特征
    df_feat = create_features(df)

    # 删除包含NaN的行（由于shift操作）
    df_feat = df_feat.dropna().reset_index(drop=True)

    print(f"特征工程后数据形状: {df_feat.shape}")

    # 特征列
    feature_cols = [
        "day_of_week",
        "days_from_start",
        "days_from_peak",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_std_7",
        "month",
        "is_weekend",
    ]

    X = df_feat[feature_cols].values
    y = df_feat["reported"].values
    dates = df_feat["Date"].values

    # 使用时序交叉验证
    tscv = TimeSeriesSplit(n_splits=5)

    # 存储结果
    oof_preds = np.zeros(len(df_feat))
    oof_preds_lower = np.zeros(len(df_feat))  # 10%分位数
    oof_preds_upper = np.zeros(len(df_feat))  # 90%分位数

    print("\n=== 时序交叉验证 ===")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        dates_val = dates[val_idx]

        print(f"\nFold {fold + 1}:")
        print(
            f"  训练集: {pd.Timestamp(dates[train_idx[0]]).date()} 到 {pd.Timestamp(dates[train_idx[-1]]).date()} ({len(train_idx)}天)"
        )
        print(
            f"  验证集: {pd.Timestamp(dates_val[0]).date()} 到 {pd.Timestamp(dates_val[-1]).date()} ({len(val_idx)}天)"
        )

        # 主模型（中位数预测）
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 5,
            "verbose": -1,
        }

        model = lgb.train(
            params,
            train_set,
            num_boost_round=300,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(-1)],
        )

        # 分位数回归模型（预测区间）
        params_lower = params.copy()
        params_lower["objective"] = "quantile"
        params_lower["alpha"] = 0.1  # 10%分位数

        params_upper = params.copy()
        params_upper["objective"] = "quantile"
        params_upper["alpha"] = 0.9  # 90%分位数

        model_lower = lgb.train(
            params_lower,
            train_set,
            num_boost_round=300,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(-1)],
        )

        model_upper = lgb.train(
            params_upper,
            train_set,
            num_boost_round=300,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(-1)],
        )

        # 预测
        oof_preds[val_idx] = model.predict(X_val)
        oof_preds_lower[val_idx] = model_lower.predict(X_val)
        oof_preds_upper[val_idx] = model_upper.predict(X_val)

        # 验证集评估
        fold_mae = mean_absolute_error(y_val, oof_preds[val_idx])
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"  MAE: {fold_mae:.2f}, RMSE: {fold_rmse:.2f}")

    # 整体评估
    print("\n=== 整体评估 ===")
    mae = mean_absolute_error(y, oof_preds)
    rmse = np.sqrt(mean_squared_error(y, oof_preds))

    # 计算各fold的平均MAE和RMSE
    fold_maes = []
    fold_rmses = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        fold_mae = mean_absolute_error(y[val_idx], oof_preds[val_idx])
        fold_rmse = np.sqrt(mean_squared_error(y[val_idx], oof_preds[val_idx]))
        fold_maes.append(fold_mae)
        fold_rmses.append(fold_rmse)

    print(f"整体 MAE: {mae:.2f}")
    print(f"整体 RMSE: {rmse:.2f}")
    print(f"各fold平均 MAE: {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f}")
    print(f"各fold平均 RMSE: {np.mean(fold_rmses):.2f} ± {np.std(fold_rmses):.2f}")

    # 预测2023年3月1日
    print("\n=== 预测2023年3月1日 ===")
    target_date = pd.Timestamp("2023-03-01")

    # 检查目标日期是否在数据范围内
    if target_date > df_feat["Date"].max():
        # 创建目标日期的特征（确保三个模型使用相同的特征值）
        new_date = target_date
        days_from_start = (new_date - df["Date"].min()).days
        days_from_peak = (
            new_date - df_feat.loc[df_feat["reported"].idxmax(), "Date"]
        ).days
        day_of_week = new_date.dayofweek
        month = new_date.month
        is_weekend = 1 if day_of_week in [5, 6] else 0

        # 使用最后7天的数据计算滚动统计
        last_7_days = df_feat["reported"].tail(7)
        rolling_mean = last_7_days.mean()
        rolling_std = last_7_days.std()

        # lag特征：使用历史数据
        lag_7 = (
            df_feat["reported"].iloc[-7]
            if len(df_feat) >= 7
            else df_feat["reported"].mean()
        )
        lag_14 = (
            df_feat["reported"].iloc[-14]
            if len(df_feat) >= 14
            else df_feat["reported"].mean()
        )

        # 构建特征向量（确保三个模型使用相同的特征）
        X_target = np.array(
            [
                [
                    day_of_week,
                    days_from_start,
                    days_from_peak,
                    lag_7,
                    lag_14,
                    rolling_mean,
                    rolling_std,
                    month,
                    is_weekend,
                ]
            ]
        )

        print(f"目标日期特征值:")
        print(f"  星期几: {day_of_week} (0=周一)")
        print(f"  距起始天数: {days_from_start}")
        print(f"  距峰值天数: {days_from_peak}")
        print(f"  前7天人数: {lag_7:.0f}")
        print(f"  前14天人数: {lag_14:.0f}")
        print(f"  过去7天均值: {rolling_mean:.0f}")
        print(f"  过去7天标准差: {rolling_std:.0f}")
        print(f"  月份: {month}")
        print(f"  是否周末: {is_weekend}")

        # 使用所有数据重新训练模型进行未来预测（而不是最后一个fold的模型）
        print("\n使用所有数据重新训练模型进行未来预测...")

        # 主模型（中位数预测）
        full_train_set = lgb.Dataset(X, label=y)

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 5,
            "verbose": -1,
        }

        full_model = lgb.train(
            params,
            full_train_set,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(-1)],
        )

        # 分位数回归模型（预测区间）
        params_lower = params.copy()
        params_lower["objective"] = "quantile"
        params_lower["alpha"] = 0.1  # 10%分位数

        params_upper = params.copy()
        params_upper["objective"] = "quantile"
        params_upper["alpha"] = 0.9  # 90%分位数

        full_model_lower = lgb.train(
            params_lower,
            full_train_set,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(-1)],
        )

        full_model_upper = lgb.train(
            params_upper,
            full_train_set,
            num_boost_round=300,
            callbacks=[lgb.log_evaluation(-1)],
        )

        # 使用相同的特征值进行预测
        pred = full_model.predict(X_target)[0]
        pred_lower = full_model_lower.predict(X_target)[0]
        pred_upper = full_model_upper.predict(X_target)[0]

        print(f"\n预测日期: {target_date.date()}")
        print(f"预测报告人数: {pred:.0f}")
        print(f"80%预测区间: [{pred_lower:.0f}, {pred_upper:.0f}]")
        print(f"区间宽度: {pred_upper - pred_lower:.0f}")

        # 验证预测值是否在区间内
        if pred_lower <= pred <= pred_upper:
            print("[OK] 预测值在预测区间内")
        else:
            print("[ERROR] 预测值不在预测区间内！需要检查模型")
            print(f"  预测值: {pred:.0f}")
            print(f"  区间下界: {pred_lower:.0f}")
            print(f"  区间上界: {pred_upper:.0f}")

        # 调试：检查三个模型的预测趋势
        print("\n[调试] 检查历史数据的预测区间:")
        # 随机选择一些历史数据点检查
        sample_indices = np.random.choice(len(X), min(10, len(X)), replace=False)
        for idx in sample_indices:
            hist_pred = full_model.predict(X[idx : idx + 1])[0]
            hist_lower = full_model_lower.predict(X[idx : idx + 1])[0]
            hist_upper = full_model_upper.predict(X[idx : idx + 1])[0]
            actual = y[idx]

            if not (hist_lower <= hist_pred <= hist_upper):
                print(
                    f"  样本{idx}: 实际值={actual:.0f}, 预测={hist_pred:.0f}, 区间=[{hist_lower:.0f}, {hist_upper:.0f}] [ERROR]"
                )
            else:
                print(
                    f"  样本{idx}: 实际值={actual:.0f}, 预测={hist_pred:.0f}, 区间=[{hist_lower:.0f}, {hist_upper:.0f}] [OK]"
                )

        # 分析未来日期特征与历史数据的差异
        print("\n[分析] 未来日期特征与历史数据对比:")
        print("  特征            未来日期值    历史平均值    历史标准差")
        for i, feat_name in enumerate(feature_cols):
            hist_mean = X[:, i].mean()
            hist_std = X[:, i].std()
            future_val = X_target[0, i]
            z_score = (future_val - hist_mean) / hist_std if hist_std > 0 else 0
            print(
                f"  {feat_name:15s} {future_val:12.1f} {hist_mean:12.1f} {hist_std:12.1f} (z={z_score:.2f})"
            )

        # 关键发现：days_from_start的z-score高达12.33，模型在外推区域
        print("\n[关键发现] days_from_start的z-score=12.33，模型在外推区域")
        print("  分位数回归在外推区域可能不稳定")
        print("  改用历史预测误差分布构建预测区间...")

        # 方法2：使用历史预测误差的分布构建预测区间
        # 计算历史预测误差
        historical_predictions = full_model.predict(X)
        errors = y - historical_predictions

        # 计算误差的分位数
        error_q10 = np.percentile(errors, 10)
        error_q90 = np.percentile(errors, 90)

        # 构建稳健的预测区间
        pred_robust_lower = pred + error_q10
        pred_robust_upper = pred + error_q90

        print(f"\n[稳健预测区间] 基于历史误差分布:")
        print(f"  预测报告人数: {pred:.0f}")
        print(f"  80%预测区间: [{pred_robust_lower:.0f}, {pred_robust_upper:.0f}]")
        print(f"  区间宽度: {pred_robust_upper - pred_robust_lower:.0f}")

        # 验证稳健区间
        if pred_robust_lower <= pred <= pred_robust_upper:
            print("  [OK] 预测值在稳健预测区间内")
        else:
            print("  [ERROR] 预测值不在稳健预测区间内")

        # 更新预测区间为稳健版本
        pred_lower = pred_robust_lower
        pred_upper = pred_robust_upper

    else:
        print(f"目标日期 {target_date.date()} 在数据范围内，使用实际值")
        target_idx = df_feat[df_feat["Date"] == target_date].index
        if len(target_idx) > 0:
            idx = target_idx[0]
            actual = y[idx]
            pred = oof_preds[idx]
            pred_lower = oof_preds_lower[idx]
            pred_upper = oof_preds_upper[idx]
            print(f"实际报告人数: {actual:.0f}")
            print(f"预测报告人数: {pred:.0f}")
            print(f"80%预测区间: [{pred_lower:.0f}, {pred_upper:.0f}]")
            print(f"预测误差: {abs(actual - pred):.0f}")

    # 可视化
    print("\n=== 生成可视化图表 ===")

    plt.figure(figsize=(14, 8))

    # 实际值
    plt.plot(dates, y, "b-", label="实际值", linewidth=2, alpha=0.7)

    # 预测值
    plt.plot(dates, oof_preds, "r-", label="预测值", linewidth=1.5, alpha=0.8)

    # 预测区间
    plt.fill_between(
        dates,
        oof_preds_lower,
        oof_preds_upper,
        color="orange",
        alpha=0.2,
        label="80%预测区间",
    )

    # 标记目标日期
    if target_date > pd.Timestamp(dates[-1]):
        plt.axvline(
            x=target_date,
            color="green",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"预测日期: {target_date.date()}",
        )
        plt.plot(target_date, pred, "go", markersize=10, label=f"预测: {pred:.0f}")
        # 确保误差条非负
        yerr_lower = max(0, pred - pred_lower)
        yerr_upper = max(0, pred_upper - pred)
        plt.errorbar(
            target_date,
            pred,
            yerr=[[yerr_lower], [yerr_upper]],
            fmt="go",
            capsize=5,
            capthick=2,
        )

    # 格式设置
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()

    plt.xlabel("日期", fontsize=12)
    plt.ylabel("报告人数", fontsize=12)
    plt.title("Wordle 报告人数时序预测 (2022-2023)", fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(OUTPUT_DIR, "q1_forecast.png")
    plt.savefig(output_path, dpi=150)
    print(f"图表已保存至: {output_path}")

    # 显示特征重要性
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    print("\n=== 特征重要性 ===")
    print(importance.to_string(index=False))

    # 特征重要性可视化
    plt.figure(figsize=(10, 6))
    plt.barh(importance["feature"], importance["importance"])
    plt.title("特征重要性 (gain)", fontsize=12)
    plt.xlabel("重要性", fontsize=10)
    plt.tight_layout()

    importance_path = os.path.join(OUTPUT_DIR, "q1_feature_importance.png")
    plt.savefig(importance_path, dpi=150)
    print(f"特征重要性图表已保存至: {importance_path}")

    print("\n模型运行完成！")


if __name__ == "__main__":
    run_model_q1()

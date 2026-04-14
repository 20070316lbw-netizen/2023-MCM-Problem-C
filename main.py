import argparse
import sys
import os

# 将 code 目录添加到路径以便导入
sys.path.append(os.path.join(os.path.dirname(__file__), "code"))

from code.eda import run_eda
from code.feature_engineering import run_feature_engineering
from code.model_q2 import run_model_q2
from code.model_q3 import run_model_q3
from code.feature_ablation import run_feature_ablation

def main():
    parser = argparse.ArgumentParser(description="2023 MCM Problem C: Wordle Analysis Pipeline")
    
    parser.add_argument("--eda", action="store_true", help="运行探索性数据分析 (EDA)")
    parser.add_argument("--fe", action="store_true", help="运行特征工程 (Feature Engineering)")
    parser.add_argument("--q2", action="store_true", help="运行问题 2 模型 (Model Q2)")
    parser.add_argument("--q3", action="store_true", help="运行问题 3 模型 (Model Q3)")
    parser.add_argument("--ablation", action="store_true", help="运行特征消融研究 (Feature Ablation)")
    parser.add_argument("--all", action="store_true", help="运行整个流水线")

    args = parser.parse_args()

    # 如果没有指定任何参数，打印帮助信息
    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.all or args.eda:
        print("\n>>> 开始运行 EDA...")
        run_eda()

    if args.all or args.fe:
        print("\n>>> 开始运行特征工程...")
        run_feature_engineering()

    if args.all or args.q2:
        print("\n>>> 开始运行问题 2 模型...")
        run_model_q2()

    if args.all or args.q3:
        print("\n>>> 开始运行问题 3 模型...")
        run_model_q3()

    if args.all or args.ablation:
        print("\n>>> 开始运行特征消融实验...")
        run_feature_ablation()

    print("\n任务完成！")

if __name__ == "__main__":
    main()

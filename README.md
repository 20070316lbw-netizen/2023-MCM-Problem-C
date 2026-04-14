# 2023 MCM Problem C: Wordle 玩家行为与单词难度分析

## 项目概述

本项目旨在分析Wordle游戏中单词特征与游戏难度之间的关系，构建一个完整的机器学习流水线来预测单词难度等级。

### 核心研究问题
1. 哪些单词特征（词频、字母组成、信息熵等）与游戏难度最相关？
2. 玩家策略（如常用开局词）如何影响游戏难度感知？
3. 能否构建一个准确的模型来预测单词的难度等级？

## 项目结构

```
2023_MCM_Problem_C/
├── data/                    # 数据文件
│   ├── Problem_C_Data_Wordle.xlsx  # 原始数据
│   ├── data_clean.csv       # 清洗后的数据
│   └── features.csv         # 特征数据
├── code/                    # 源代码
│   ├── __init__.py
│   ├── load_data.py         # 数据加载
│   ├── eda.py              # 探索性数据分析
│   ├── feature_engineering.py  # 特征工程
│   ├── difficulty_label.py  # 难度标签定义
│   ├── model_q2.py         # 问题2模型
│   ├── model_q3.py         # 问题3模型
│   ├── feature_ablation.py  # 特征消融实验
│   └── visualization.py    # 可视化模块
├── output/                  # 输出目录
│   └── figures/            # 生成的图表
├── main.py                 # 主程序入口
├── main.ipynb              # Jupyter分析报告
├── main.tex               # LaTeX论文模板
├── requirements.txt        # 依赖包列表
├── setup_environment.py    # 环境设置脚本
└── README.md              # 本文件
```

## 环境设置

### 方法1：使用设置脚本（推荐）
```bash
# 运行环境设置脚本
python setup_environment.py
```

### 方法2：手动安装
```bash
# 安装依赖包
pip install -r requirements.txt

# 安装NLTK数据
python -c "import nltk; nltk.download('words')"
```

### 方法3：使用conda
```bash
# 创建conda环境
conda create -n wordle_analysis python=3.9
conda activate wordle_analysis

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 1. 运行完整分析流水线
```bash
python main.py --all
```

### 2. 运行特定模块
```bash
# 探索性数据分析
python main.py --eda

# 特征工程
python main.py --fe

# 问题2模型
python main.py --q2

# 问题3模型
python main.py --q3

# 特征消融实验
python main.py --ablation

# 可视化
python main.py --viz
```

### 3. 查看帮助
```bash
python main.py --help
```

### 4. 使用Jupyter Notebook
```bash
# 启动Jupyter
jupyter notebook

# 打开 main.ipynb 查看完整分析报告
```

## 可视化模块

可视化模块 (`code/visualization.py`) 生成6个关键图表：

1. **特征相关性热力图** (`output/figures/feature_correlation.png`)
2. **特征散点图** (`output/figures/feature_scatter.png`)
3. **难度箱线图** (`output/figures/difficulty_boxplot.png`)
4. **混淆矩阵** (`output/figures/confusion_matrix.png`)
5. **单词分布对比图** (`output/figures/distribution_comparison.png`)
6. **Q2 MAE对比图** (`output/figures/q2_mae.png`)

运行可视化：
```bash
python main.py --viz
# 或
python code/visualization.py
```

## 技术细节

### 特征工程
- **语言学特征**: 词频、信息熵、重复字母数
- **策略特征**: 与常用开局词的字母重叠度
- **结构特征**: 首字母候选空间
- **时间特征**: 归一化的Hard Mode比例

### 模型构建
- **算法**: LightGBM分类器
- **评估**: 5折交叉验证
- **目标**: 三分类（简单、中等、困难）

### 数据处理
- 异常值处理：剔除2022-11-30（单词EERIE）的异常数据
- 难度量化：基于平均尝试次数的分位数划分
- 特征归一化：时间序列特征的滚动中位数归一化

## 主要发现

1. **词频与难度负相关**: 常见单词更容易猜中
2. **开局词策略有效性**: 与常用开局词重叠度高的单词更容易被猜中
3. **字母重复增加难度**: 有重复字母的单词更难猜
4. **首字母搜索空间影响**: 以常见字母开头的单词搜索空间更大，增加难度

## 依赖包

- pandas, numpy: 数据处理
- matplotlib, seaborn: 数据可视化
- scikit-learn, lightgbm: 机器学习
- nltk, wordfreq: 自然语言处理
- openpyxl: Excel文件支持

完整列表见 `requirements.txt`

## 许可证

本项目仅供学习和研究使用。

## 作者

2023 MCM Problem C 分析项目

## 更新日志

- 2024-04-14: 创建项目，添加可视化模块和完整分析流水线
- 2024-04-14: 更新main.ipynb，添加详细原理说明
- 2024-04-14: 添加环境设置脚本和依赖管理

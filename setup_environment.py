#!/usr/bin/env python
"""
环境设置脚本
用于安装项目所需的所有依赖包
"""

import sys
import subprocess
import importlib.util


def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    if sys.version_info < (3, 8):
        print(f"错误: 需要Python 3.8或更高版本，当前版本: {sys.version}")
        return False
    print(f"OK Python版本: {sys.version}")
    return True


def install_package(package, pip_name=None):
    """安装单个包"""
    if pip_name is None:
        pip_name = package

    try:
        # 检查包是否已安装
        spec = importlib.util.find_spec(package)
        if spec is not None:
            print(f"  OK {package} 已安装")
            return True
    except ImportError:
        pass

    print(f"  正在安装 {package}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "--quiet"]
        )
        print(f"  OK {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR {package} 安装失败: {e}")
        return False


def install_nltk_data():
    """安装NLTK数据"""
    print("\n安装NLTK数据...")
    try:
        import nltk

        nltk.download("words", quiet=True)
        print("  OK NLTK words corpus 安装成功")
        return True
    except Exception as e:
        print(f"  ERROR NLTK数据安装失败: {e}")
        return False


def main():
    print("=" * 60)
    print("2023 MCM Problem C: Wordle Analysis 环境设置")
    print("=" * 60)

    # 检查Python版本
    if not check_python_version():
        return

    # 更新pip
    print("\n更新pip...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"]
        )
        print("  OK pip更新成功")
    except subprocess.CalledProcessError:
        print("  WARNING pip更新失败，继续安装...")

    # 安装核心包
    print("\n安装核心依赖包...")
    packages = [
        ("pandas", None),
        ("numpy", None),
        ("matplotlib", None),
        ("seaborn", None),
        ("sklearn", "scikit-learn"),
        ("lightgbm", None),
        ("wordfreq", None),
        ("nltk", None),
        ("openpyxl", None),
    ]

    all_installed = True
    for package, pip_name in packages:
        if not install_package(package, pip_name):
            all_installed = False

    # 安装NLTK数据
    if all_installed:
        install_nltk_data()

    print("\n" + "=" * 60)
    if all_installed:
        print("环境设置完成！")
        print("\n可以运行以下命令测试：")
        print("  python main.py --help          # 查看帮助")
        print("  python main.py --viz           # 运行可视化")
        print("  python main.py --all           # 运行完整流水线")
        print("  python code/visualization.py   # 单独运行可视化模块")
    else:
        print("环境设置过程中遇到问题，请手动安装缺失的包。")
        print("运行: pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()

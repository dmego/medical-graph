"""
公共的中文图表样式配置模块

提供统一的 matplotlib/seaborn 中文显示配置，
用于所有分析模块的图表绘制。
"""

import matplotlib.pyplot as plt
import seaborn as sns


def configure_chinese_plot_style() -> None:
    """
    配置 matplotlib/seaborn 样式以支持中文显示。

    设置包括：
    - seaborn 主题样式
    - 中文字体候选列表（支持 macOS、Windows、Linux）
    - Unicode 负号显示修复
    - 字体回退机制

    使用方法：
        from src.set_chinese_plot import configure_chinese_plot_style
        configure_chinese_plot_style()
    """
    # 设置 seaborn 主题
    sns.set_theme(style="whitegrid")

    # 中文字体候选列表，按优先级排序
    font_candidates = [
        "PingFang HK",      # macOS 中文
        "PingFang SC",      # macOS 简体中文
        "STHeiti",          # macOS 黑体
        "Heiti TC",         # macOS 繁体黑体
        "Hei",              # 系统黑体
        "SimHei",           # Windows 黑体
        "Microsoft YaHei",  # Windows 微软雅黑
        "Kaiti SC",         # 楷体简体
        "SimSong",          # 宋体
        "WenQuanYi Micro Hei",  # Linux 文泉驿微米黑
        "WenQuanYi Zen Hei",    # Linux 文泉驿正黑
        "DejaVu Sans",      # 通用无衬线字体
        "Arial Unicode MS", # 跨平台 Unicode 字体
        "sans-serif",       # 默认无衬线字体
    ]

    # 设置字体
    plt.rcParams["font.sans-serif"] = font_candidates
    plt.rcParams["font.family"] = "sans-serif"

    # 修复负号显示问题（使用 ASCII 负号而非 Unicode）
    plt.rcParams["axes.unicode_minus"] = False

    # 设置字体大小和样式
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9

    # 禁用字体缺失警告（减少控制台输出）
    import warnings
    warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*", category=UserWarning)


# 为了向后兼容，提供别名
setup_chinese_plot = configure_chinese_plot_style
configure_plot_style = configure_chinese_plot_style
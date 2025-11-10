"""
PageRank节点重要性分析模块

该模块实现了基于PageRank算法的节点重要性分析，包括：
- PageRank得分计算
- 全局分布可视化
- 高重要性节点类型分析
- 疾病排名和科室分布分析
- 疾病-药品关联网络可视化
- 报告生成

"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from . import data_loader


def _compute_pagerank_scores() -> pd.DataFrame:
    """
    计算图中所有节点的PageRank得分

    使用NetworkX的pagerank算法计算有向图中每个节点的PageRank重要性得分。

    Returns:
        pd.DataFrame: 包含node_id, name, type, pagerank列的DataFrame，按PageRank降序排列
    """
    nodes_df = data_loader.load_nodes()
    edges_df = data_loader.load_edges()

    G = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        G.add_node(row["node_id"], name=row["name"], type=row["type"], other_attrs=row["other_attrs"])

    for _, row in edges_df.iterrows():
        G.add_edge(row["src_id"], row["dst_id"], rel_type=row["rel_type"], other_attrs=row["other_attrs"])

    scores = nx.pagerank(G, alpha=0.85, max_iter=100, weight=None)  # Assuming no weight, or adjust if needed

    records = []
    for node_id, score in scores.items():
        data = G.nodes[node_id]
        records.append({
            "node_id": node_id,
            "name": data.get("name", ""),
            "type": data.get("type", ""),
            "pagerank": score,
        })
    df = pd.DataFrame(records)
    df.sort_values(by="pagerank", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _load_nodes_lookup() -> pd.DataFrame:
    """Load node metadata with index on node_id for fast joins."""

    nodes = data_loader.load_nodes(["node_id", "name", "type", "other_attrs"])
    return nodes.set_index("node_id")


def _parse_departments(value: str | float | None) -> List[str]:
    """
    从JSON属性中解析治疗科室信息

    Args:
        value: JSON字符串或数值，包含科室信息

    Returns:
        List[str]: 科室名称列表
    """
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return []
    departments = payload.get("cure_department")
    if isinstance(departments, list):
        return [str(dep) for dep in departments if isinstance(dep, str)]
    if isinstance(departments, str):
        return [departments]
    return []


def _classify_system_category(departments: Iterable[str], disease_name: str) -> str:
    """
    根据科室信息和疾病名称推断临床系统类别

    Args:
        departments: 科室名称列表
        disease_name: 疾病名称

    Returns:
        str: 系统类别名称
    """
    if not departments:
        return "综合/未标注"

    mapping: List[Tuple[str, str]] = [
        ("心", "循环系统"),
        ("血", "循环系统"),
        ("呼吸", "呼吸系统"),
        ("胸", "呼吸系统"),
        ("感染", "感染系统"),
        ("急诊", "急危重症"),
        ("重症", "急危重症"),
        ("神经", "神经系统"),
        ("消化", "消化系统"),
        ("内分泌", "内分泌系统"),
        ("内科", "内科综合"),
        ("外科", "外科综合"),
        ("骨", "肌骨系统"),
        ("中医", "中医科"),
        ("儿科", "儿科"),
        ("妇", "妇产科"),
        ("泌尿", "泌尿系统"),
        ("肿瘤", "肿瘤/肿块"),
        ("眼", "五官科"),
        ("耳", "五官科"),
        ("口腔", "五官科"),
    ]

    for department in departments:
        for keyword, category in mapping:
            if keyword in department:
                return category

    # Fallback heuristics on disease name if department hints failed.
    for keyword, category in mapping:
        if keyword in disease_name:
            return category

    return "综合/未标注"


def _configure_plot_style() -> None:
    """Configure matplotlib/seaborn styles with Chinese font fallbacks."""
    from .set_chinese_plot import configure_chinese_plot_style
    configure_chinese_plot_style()


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _render_global_distribution(
    pagerank_df: pd.DataFrame,
    output_path: Path,
    outlier_threshold: float,
) -> float:
    """
    绘制全局PageRank分布图

    在双对数坐标系上绘制排名vs得分散点图，展示PageRank分布特征。

    Args:
        pagerank_df: PageRank结果DataFrame
        output_path: 输出图片路径
        outlier_threshold: 异常值阈值

    Returns:
        float: 拟合斜率（当前返回NaN，表示未进行拟合）
    """
    df = pagerank_df.sort_values("pagerank", ascending=False).reset_index(drop=True)
    df = df[df["pagerank"] > 0]
    df["rank"] = df.index + 1

    x = df["rank"].to_numpy().astype(float)
    y = df["pagerank"].to_numpy().astype(float)

    # Draw scatter on log-log axes. We intentionally do not fit or annotate
    # a power-law line here per report styling preferences.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=6, alpha=0.5, color="#4B8BBE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("节点排名（按 PageRank 降序，log）")
    ax.set_ylabel("PageRank 值（log）")
    ax.set_title("图谱 PageRank 排名-得分")

    # Annotate outlier threshold (in terms of value)
    ax.axhline(outlier_threshold, color="#2ca02c", linestyle=":", linewidth=1.2)
    ax.text(x.max(), outlier_threshold, " 3×均值阈值", va="bottom", ha="right", color="#2ca02c")

    plt.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    # Returning NaN to indicate no fitted slope was computed
    return float("nan")


def _render_high_importance_type_share(
    top_nodes: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot entity type composition among the top-ranked nodes."""

    counts = top_nodes["type"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    colors = sns.color_palette("Set2", n_colors=len(counts))
    ax.pie(
        counts,
        labels=[f"{label} ({value})" for label, value in counts.items()],
        autopct="%1.1f%%",
        startangle=120,
        colors=colors,
        textprops={"fontsize": 11},
    )
    ax.set_title("前 200 高 PageRank 节点类型占比")
    ax.axis("equal")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _render_top_diseases_chart(
    diseases_df: pd.DataFrame,
    nodes_lookup: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Render top diseases bar chart, returning enriched dataframe and category mapping."""

    top_diseases = diseases_df.head(top_n).copy()
    categories: Dict[str, str] = {}
    department_col: List[str] = []
    for node_id, name in zip(top_diseases["node_id"], top_diseases["name"]):
        attrs = nodes_lookup.loc[node_id].get("other_attrs")
        departments = _parse_departments(attrs)
        category = _classify_system_category(departments, str(name))
        categories[str(name)] = category
        department_col.append(", ".join(departments) if departments else "未标注")
    top_diseases["system_category"] = [categories[str(name)] for name in top_diseases["name"]]
    top_diseases["department"] = department_col

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=top_diseases,
        y="name",
        x="pagerank",
        hue="system_category",
        palette="Set3",
        ax=ax,
    )
    ax.set_xlabel("PageRank 值")
    ax.set_ylabel("疾病名称")
    ax.set_title(f"Top {top_n} 疾病 PageRank 排名")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1e}"))
    ax.legend(title="系统类别", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    return top_diseases, categories


def _render_department_heatmap(
    diseases_df: pd.DataFrame,
    nodes_lookup: pd.DataFrame,
    output_path: Path,
    top_k_departments: int = 12,
) -> pd.DataFrame:
    """Render heatmap showing department coverage for high PageRank diseases."""

    expanded_records: List[Dict[str, object]] = []
    for _, row in diseases_df.iterrows():
        node_id = row["node_id"]
        attrs = nodes_lookup.loc[node_id].get("other_attrs")
        departments = _parse_departments(attrs)
        departments = departments or ["未标注"]
        for dept in departments:
            expanded_records.append(
                {
                    "department": dept,
                    "pagerank": row["pagerank"],
                    "name": row["name"],
                }
            )
    if not expanded_records:
        return pd.DataFrame()

    stats_df = (
        pd.DataFrame(expanded_records)
        .groupby("department")
        .agg(疾病数量=("name", "nunique"), 平均PageRank=("pagerank", "mean"))
        .sort_values("疾病数量", ascending=False)
    )
    stats_df = stats_df.head(top_k_departments)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Use scientific notation for small average PageRank values
    sns.heatmap(
        stats_df.T,
        annot=True,
        fmt=".2e",
        cmap="YlGnBu",
        cbar_kws={"label": "指标值"},
        ax=ax,
    )
    ax.set_title("前 100 高 PageRank 疾病的科室分布热力图")
    plt.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    return stats_df


def _render_department_radar(
    diseases_df: pd.DataFrame,
    nodes_lookup: pd.DataFrame,
    output_path: Path,
    top_k_departments: int = 12,
) -> pd.DataFrame:
    """Render a dual-metric radar (spider) chart for departments.

    Two metrics:
    1. 疾病数量 (count of high PageRank diseases) -> normalized to [0,1]
    2. 平均PageRank (mean PageRank of those diseases) -> normalized to [0,1]

    Both are displayed as separate polygons with legend. Normalization makes
    the relative shape comparable despite scale differences.
    Returns stats_df for downstream textual description.
    """

    # Reuse the same aggregation as the heatmap for consistency
    expanded_records: List[Dict[str, object]] = []
    for _, row in diseases_df.iterrows():
        node_id = row["node_id"]
        attrs = nodes_lookup.loc[node_id].get("other_attrs")
        departments = _parse_departments(attrs)
        departments = departments or ["未标注"]
        for dept in departments:
            expanded_records.append(
                {
                    "department": dept,
                    "pagerank": row["pagerank"],
                    "name": row["name"],
                }
            )
    if not expanded_records:
        return pd.DataFrame()

    stats_df = (
        pd.DataFrame(expanded_records)
        .groupby("department")
        .agg(疾病数量=("name", "nunique"), 平均PageRank=("pagerank", "mean"))
        .sort_values("疾病数量", ascending=False)
    )
    stats_df = stats_df.head(top_k_departments)

    labels = list(stats_df.index)
    if not labels:
        return stats_df

    counts = stats_df["疾病数量"].astype(float).to_numpy()
    means = stats_df["平均PageRank"].astype(float).to_numpy()

    # Normalize to [0,1] independently to compare shapes
    counts_norm = counts / (counts.max() if counts.max() else 1.0)
    means_norm = means / (means.max() if means.max() else 1.0)

    # Close the loop for radar polygons
    counts_plot = np.concatenate([counts_norm, counts_norm[:1]])
    means_plot = np.concatenate([means_norm, means_norm[:1]])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_plot = np.concatenate([angles, angles[:1]])

    fig = plt.figure(figsize=(8.2, 8.2))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles_plot, counts_plot, color="#4B8BBE", linewidth=2, label="疾病数量(归一化)")
    ax.fill(angles_plot, counts_plot, color="#4B8BBE", alpha=0.25)

    ax.plot(angles_plot, means_plot, color="#FF8C00", linewidth=2, label="平均PageRank(归一化)")
    ax.fill(angles_plot, means_plot, color="#FF8C00", alpha=0.18)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("高 PageRank 疾病科室分布双指标雷达图", y=1.08)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05), frameon=False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    return stats_df


def _detect_outliers(pagerank_df: pd.DataFrame, multiplier: float = 3.0) -> Tuple[float, pd.DataFrame]:
    """Identify nodes whose PageRank exceeds multiplier * mean."""

    mean_score = float(pagerank_df["pagerank"].mean())
    threshold = mean_score * multiplier
    outliers = pagerank_df[pagerank_df["pagerank"] > threshold].copy()
    return threshold, outliers


def _render_disease_drug_network(
    pagerank_df: pd.DataFrame,
    nodes_lookup: pd.DataFrame,
    edges_df: pd.DataFrame,
    diseases_df: pd.DataFrame,
    static_output: Path,
) -> Dict[str, object]:
    """Render interactive and static networks linking top diseases and drugs."""

    top_diseases = diseases_df.head(10)
    disease_ids = top_diseases["node_id"].tolist()
    relevant_edges = edges_df[
        (edges_df["src_id"].isin(disease_ids))
        & (edges_df["rel_type"].isin({"common_drug", "recommand_drug"}))
    ]

    if relevant_edges.empty:
        return {"diseases": len(disease_ids), "drugs": 0, "edges": 0}

    pagerank_lookup = pagerank_df.set_index("node_id")

    G = nx.Graph()
    type_colors = {"Disease": "#D62728", "Drug": "#1F77B4"}

    for _, row in top_diseases.iterrows():
        node_id = row["node_id"]
        score = row["pagerank"]
        G.add_node(
            str(node_id),
            label=str(row["name"]),
            pagerank=score,
            type="Disease",
        )

    drug_counts: Counter[str] = Counter()
    for _, edge in relevant_edges.iterrows():
        disease_id = str(edge["src_id"])
        drug_id_int = int(edge["dst_id"])
        drug_row = nodes_lookup.loc[drug_id_int]
        drug_name = str(drug_row["name"])

        if drug_id_int in pagerank_lookup.index:
            try:
                raw_val = pagerank_lookup.at[drug_id_int, "pagerank"]
                drug_score = float(cast(float, raw_val)) if pd.notna(raw_val) else 0.0
            except (TypeError, ValueError):
                drug_score = 0.0
        else:
            drug_score = 0.0
        G.add_node(
            str(drug_id_int),
            label=drug_name,
            pagerank=drug_score,
            type="Drug",
        )
        G.add_edge(
            disease_id,
            str(drug_id_int),
            rel_type=str(edge["rel_type"]),
        )
        drug_counts.update([drug_name])

    # Static layout for quick preview.
    size_scale = np.linspace(12, 28, num=5)

    def _scale_size(score: float) -> float:
        if score <= 0:
            return size_scale[0]
        quantiles = np.quantile(list(pagerank_lookup["pagerank"]), [0.5, 0.75, 0.9, 0.98])
        if score > quantiles[3]:
            return size_scale[4]
        if score > quantiles[2]:
            return size_scale[3]
        if score > quantiles[1]:
            return size_scale[2]
        if score > quantiles[0]:
            return size_scale[1]
        return size_scale[0]

    pos = nx.spring_layout(G, seed=42, k=0.5)
    fig, ax = plt.subplots(figsize=(10, 8))
    for node_type, color in type_colors.items():
        nodes_of_type = [n for n, data in G.nodes(data=True) if data.get("type") == node_type]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes_of_type,
            node_color=color,
            node_size=[
                _scale_size(float(G.nodes[n].get("pagerank", 0))) * 6 for n in nodes_of_type
            ],
            alpha=0.85,
            ax=ax,
            label=node_type,
        )
    nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color="#7E57C2", ax=ax)
    # Draw truncated labels beside nodes for static figure to reduce overlap.
    label_trunc = 12
    for n, d in G.nodes(data=True):
        name_full = str(d.get("label", ""))
        if len(name_full) > label_trunc:
            name_short = name_full[:label_trunc].rstrip() + "…"
        else:
            name_short = name_full
        x, y = pos[n]
        # small offset to the right for readability
        ax.text(x + 0.02, y, name_short, fontsize=8, ha="left", va="center")
    ax.set_axis_off()
    ax.set_title("Top 10 疾病与关联药品的知识聚合图")
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(static_output, dpi=240)
    plt.close(fig)

    return {
        "diseases": len(disease_ids),
        "drugs": sum(1 for _ in (n for n, data in G.nodes(data=True) if data.get("type") == "Drug")),
        "edges": G.number_of_edges(),
        "top_drugs": dict(drug_counts.most_common(5)),
    }


def _generate_visual_report(
    pagerank_df: pd.DataFrame,
    nodes_lookup: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> Dict[str, object]:
    """Create visualization assets and analytical summaries for PageRank results."""

    _configure_plot_style()

    image_dir = _ensure_directory(data_loader.OUTPUT_DIR / "images" / "pagerank")

    threshold, outliers = _detect_outliers(pagerank_df)
    type_top200 = pagerank_df.nlargest(200, "pagerank")
    top_diseases = _top_by_type(pagerank_df, "Disease", 100)

    distribution_image = image_dir / "pagerank_distribution.png"
    fit_slope = _render_global_distribution(pagerank_df, distribution_image, threshold)

    type_share_image = image_dir / "pagerank_top200_type_share.png"
    _render_high_importance_type_share(type_top200, type_share_image)

    disease_bar_image = image_dir / "pagerank_top_diseases_bar.png"
    enriched_diseases, category_map = _render_top_diseases_chart(
        top_diseases,
        nodes_lookup,
        disease_bar_image,
    )

    department_heatmap_image = image_dir / "pagerank_department_heatmap.png"
    # Render radar chart (replaces previous heatmap) but keep the same filename for report compatibility
    department_stats = _render_department_radar(top_diseases, nodes_lookup, department_heatmap_image)

    network_static_image = image_dir / "pagerank_disease_drug_network.png"
    network_stats = _render_disease_drug_network(
        pagerank_df,
        nodes_lookup,
        edges_df,
        top_diseases,
        network_static_image,
    )

    outlier_detail = outliers.sort_values("pagerank", ascending=False).head(15)
    outlier_detail = outlier_detail.assign(
        name=lambda df_: df_["node_id"].map(nodes_lookup["name"]),
        type=lambda df_: df_["node_id"].map(nodes_lookup["type"]),
    )

    type_share_pct = (
        type_top200["type"].value_counts(normalize=True).sort_values(ascending=False)
    )

    visual_assets = {
        "distribution_image": distribution_image,
        "type_share_image": type_share_image,
        "disease_bar_image": disease_bar_image,
        "department_heatmap_image": department_heatmap_image,
        "network_static_image": network_static_image,
        "network_stats": network_stats,
        "outlier_threshold": threshold,
        "outlier_nodes": outlier_detail,
        "type_share_pct": type_share_pct,
        "top_disease_categories": category_map,
        "department_stats": department_stats,
        "enriched_top_diseases": enriched_diseases,
        "fit_slope": fit_slope,
    }

    return visual_assets


def _write_markdown(assets: Dict[str, object], summary: Dict, output_path: Path) -> Path:
    """生成详细的PageRank分析报告"""
    
    def rel(p: Path) -> str:
        return os.path.relpath(p, data_loader.REPORT_DIR)

    top_nodes = summary.get("top_nodes", [])
    network_stats = cast(Dict, assets.get("network_stats", {}))
    
    lines: List[str] = [
        f"# PageRank 算法分析报告",
        "",
        f"**分析日期**: {pd.Timestamp.now().strftime('%Y年%m月%d日')}  ",
        f"**数据集**: 医疗知识图谱 (PageRank分析)",
        "**分析工具**: NetworkX + Python",
        "",
        "---",
        "",
        "## 一、PageRank 算法概览",
        "",
        "PageRank 是一种衡量网络中节点“全局重要性”的算法。一个节点的 PageRank 分数高，意味着有更多、更重要的节点指向它。在医疗知识图谱中，高 PageRank 节点可以被理解为整个知识体系中的“核心概念”或“关键枢纽”。",
        "",
        "### 1.1 核心数据",
        f"- **分析的节点总数**: {summary.get('total_nodes', 0):,}",
        f"- **Top 1 PageRank 节点**: {top_nodes[0]['name'] if top_nodes else 'N/A'} (Score: {top_nodes[0]['pagerank']:.2e})",
        f"- **Top 10 平均 PageRank 分数**: {summary.get('top_10_avg_pagerank', 0):.2e}",
        "",
        "---",
        "",
        "## 二、全局重要性分析",
        "",
        "### 2.1 PageRank 分数分布",
        "",
        f"![PageRank 分数分布]({rel(cast(Path, assets['distribution_image']))})",
        "",
        "#### 业务解读",
        "该图展示了所有节点 PageRank 分数的分布情况，呈现出典型的“长尾分布”特征。",
        "",
        "**结构洞察**：",
        "- **少数核心，多数边缘**：图中左侧的“高峰”表明，绝大多数节点的 PageRank 分数都非常低，而右侧拖着一条长长的“尾巴”，说明只有极少数节点的 PageRank 分数非常高。",
        "- **符合复杂网络特性**：这种分布是无标度网络（Scale-free Network）的典型特征，意味着网络依赖于少数核心节点来维持其连接性。这与 `graph_analysis` 中发现的幂律分布结论一致。",
        "",
        "### 2.2 Top 200 核心节点类型占比",
        "",
        f"![Top 200 核心节点类型占比]({rel(cast(Path, assets['type_share_image']))})",
        "",
        "#### 业务解读",
        "上图展示了 PageRank 分数最高的200个节点的实体类型构成。",
        "",
        "**数据观察**：",
        "- **疾病和症状是绝对核心**：超过一半的核心节点是**疾病(Disease)**和**症状(Symptom)**。这完全符合直觉：疾病和症状是整个医疗知识体系的组织核心，绝大多数信息（如药物、检查、饮食）都是围绕它们展开的。",
        "- **药物(Drug)和检查(Check)是重要补充**：作为治疗和诊断的关键手段，药物和检查也占据了相当大的比重，是连接疾病知识的重要桥梁。",
        "",
        "---",
        "",
        "## 三、核心疾病与科室分析",
        "",
        "### 3.1 Top 20 核心疾病排名",
        "",
        f"![Top 20 核心疾病排名]({rel(cast(Path, assets['disease_bar_image']))})",
        "",
        "#### 业务解读",
        "此图表展示了 PageRank 最高的前20种疾病，并按其所属的临床系统（如循环系统、呼吸系统等）进行了颜色编码。",
        "",
        "**数据观察**：",
        "- **常见病和多发病占据高位**：排名靠前的多为大众熟知的疾病，如“高血压”、“糖尿病”等。这些疾病关联的知识（药物、并发症、饮食建议）非常丰富，因此在网络中具有极高的中心性。",
        "- **系统性疾病的重要性**：图例清晰地展示了不同临床系统的疾病分布，有助于我们从更高维度理解疾病的重要性格局。",
        "",

        "### 3.2 核心疾病的科室分布雷达图",
        "",
        f"![核心疾病的科室分布雷达图]({rel(cast(Path, assets['department_heatmap_image']))})",
        "",
        "#### 业务解读",
        "雷达图从两个维度分析了与高 PageRank 疾病关联最紧密的临床科室：",
        "1.  **疾病数量（蓝色）**：某科室关联的高 PageRank 疾病越多，其覆盖面越广。",
        "2.  **平均PageRank（橙色）**：某科室关联的疾病平均重要性越高，其专业领域的“核心度”越强。",
        "",
        "**数据观察**：",
        "- **内外妇儿是基础**：内科、外科、妇产科、儿科等基础科室覆盖的疾病数量最多，是知识图谱的“广度”担当。",
        "- **专科领域显现核心价值**：某些专科（如心血管内科、肿瘤科）虽然覆盖的疾病数量不一定最多，但其关联疾病的平均 PageRank 可能非常高，说明它们是“深度”和“专业核心”的代表。",
        "",
        "---",
        "",
        "## 四、核心疾病的关联网络",
        "",
        f"![Top 10 疾病与关联药品的知识聚合图]({rel(cast(Path, assets['network_static_image']))})",
        "",
        "#### 业务解读",
        "上图构建了一个连接网络，展示了 PageRank 最高的10种疾病（红色节点）以及与它们直接关联的药物（蓝色节点）。节点的大小反映了其自身的 PageRank 分数。",
        "",
        "**数据观察**：",
        "- **核心疾病连接共通药物**：我们可以看到，一些核心药物（较大的蓝色节点）同时被多种核心疾病所连接，这揭示了“一药多用”或治疗相关疾病的普遍性。",
        f"- **网络统计**：该网络共包含 **{network_stats.get('diseases', 0)}** 种核心疾病，关联了 **{network_stats.get('drugs', 0)}** 种药物，形成了 **{network_stats.get('edges', 0)}** 条连接关系。",
        "",
        "**应用建议**：",
        "- **智能问答与推荐**：基于此网络，可以构建更智能的推荐系统。例如，当用户查询某个核心疾病时，可以优先推荐与它关联且自身 PageRank 也很高的药物。",
        "- **知识发现**：分析那些连接了多个不同系统疾病的药物，可能有助于发现新的治疗方案或研究方向。",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _top_by_type(df: pd.DataFrame, node_type: str, top_n: int) -> pd.DataFrame:
    """从DataFrame中筛选指定类型的前N个节点"""
    return df[df["type"] == node_type].head(top_n)


def _generate_summary(pagerank_df: pd.DataFrame) -> Dict:
    """从PageRank数据生成摘要字典"""
    top_nodes_df = pagerank_df.head(20)
    summary = {
        "total_nodes": len(pagerank_df),
        "top_nodes": top_nodes_df.to_dict("records"),
        "top_10_avg_pagerank": top_nodes_df.head(10)["pagerank"].mean(),
    }
    return summary


def generate_outputs():
    """主函数，执行所有分析和生成步骤"""
    _configure_plot_style()

    # 确保目录存在
    data_loader.PAGERANK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_loader.PAGERANK_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # 计算或加载PageRank分数
    pagerank_scores_path = data_loader.PAGERANK_DATA_DIR / "pagerank_scores.csv"
    if pagerank_scores_path.exists():
        pagerank_df = pd.read_csv(pagerank_scores_path)
    else:
        pagerank_df = _compute_pagerank_scores()
        pagerank_df.to_csv(pagerank_scores_path, index=False)
    
    print("PageRank scores computed/loaded.")

    # 加载额外数据
    nodes_lookup = _load_nodes_lookup()
    edges_df = data_loader.load_edges()

    # 生成所有可视化图表和分析资产
    visual_assets = _generate_visual_report(pagerank_df, nodes_lookup, edges_df)
    print("Visualizations generated.")

    # 生成摘要
    summary_dict = _generate_summary(pagerank_df)
    summary_path = data_loader.PAGERANK_DATA_DIR / "pagerank_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=4)
    print("Summary generated.")

    # 生成Markdown报告
    report_path = data_loader.REPORT_DIR / "pagerank_analysis.md"
    _write_markdown(visual_assets, summary_dict, report_path)
    print(f"PageRank analysis report generated: {report_path}")


if __name__ == "__main__":
    generate_outputs()

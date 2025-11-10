"""
Louvain社区发现分析模块

该模块实现了基于Louvain算法的社区发现分析，包括：
- 社区划分计算
- 社区级图构建
- 可视化分析
- 报告生成

"""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from community import community_louvain
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter

from . import data_loader

try:
    from wordcloud import WordCloud  # type: ignore
except Exception:  # pragma: no cover - 环境缺失时兜底
    WordCloud = None  # noqa: N816

try:
    import jieba  # type: ignore
except Exception:  # pragma: no cover
    jieba = None


# ----------------------------- Style & IO -----------------------------
def _configure_style() -> None:
    from .set_chinese_plot import configure_chinese_plot_style
    configure_chinese_plot_style()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_community_titles() -> Dict[int, str]:
    """从 data/louvain/louvain_top10_name.txt 中读取社区名称映射；若不存在则返回空映射。"""
    mapping: Dict[int, str] = {}
    txt_path = data_loader.resolve_output("data/louvain/louvain_top10_name.txt")
    if not txt_path.exists():
        return mapping
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.strip().split('\n'):
        if ':' in line:
            parts = line.split(':', 1)
            try:
                cid = int(parts[0].strip())
                title = parts[1].strip()
                mapping[cid] = title
            except ValueError:
                continue
    return mapping


def _auto_community_titles(assign_df: pd.DataFrame, cids: List[int]) -> Dict[int, str]:
    """基于社区内节点名称的高频词，自动生成社区标题（中文分词优先）。
    规则：
    - 仅使用 Disease/Symptom/Check 名称进行分词统计，更贴近医学主题词；
    - 过滤停用词与长度<2 的 token；
    - 取前2个高频词，组合为“词1/词2 主题群”；
    - 若分词不可用或无词，回退为主类型组合如“疾病-症状 主题群”。
    """
    stop = set(["综合征", "病", "症", "性", "相关", "伴", "的", "与", "及", "并", "或", "型", "急性", "慢性", "检查", "治疗"])  # 简易停用词
    title_map: Dict[int, str] = {}
    for cid in cids:
        sub = assign_df[(assign_df["community"] == cid) & (assign_df["type"].isin(["Disease", "Symptom", "Check"]))]
        names = sub["name"].astype(str).tolist()
        tokens: List[str] = []
        if jieba is not None:
            for n in names:
                toks = [t for t in jieba.lcut(n) if len(t) >= 2 and t not in stop and not t.isdigit()]
                tokens.extend(toks)
        else:
            # 无分词时按非字母数字分割
            import re
            for n in names:
                toks = [t for t in re.split(r"[^\u4e00-\u9fa5A-Za-z]+", n) if len(t) >= 2 and t not in stop]
                tokens.extend(toks)
        if tokens:
            vc = pd.Series(tokens).value_counts()
            words = vc.index.tolist()[:2]
            title_map[cid] = "/".join(words) + " 主题群"
        else:
            # 回退：主类型前两名
            td = assign_df[assign_df["community"] == cid]["type"].value_counts().index.tolist()[:2]
            title_map[cid] = ("-".join(td) if td else "社区主题") + " 主题群"
    return title_map


# ----------------------------- Data Loading -----------------------------
def _compute_louvain() -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    执行Louvain社区发现算法

    该函数加载节点和边数据，构建图结构，运行Louvain算法进行社区划分，
    并计算社区统计信息。

    Returns:
        Tuple[pd.DataFrame, Dict[str, object]]: 社区划分结果DataFrame和统计摘要
            - DataFrame包含node_id, community, name, type列
            - Dict包含num_communities, modularity, largest_community_size, community_insights
    """
    nodes_df = data_loader.load_nodes()
    edges_df = data_loader.load_edges()

    # 构建多重有向图
    graph = nx.MultiDiGraph()
    for _, row in nodes_df.iterrows():
        graph.add_node(int(row["node_id"]), name=row["name"], type=row["type"])

    for _, row in edges_df.iterrows():
        graph.add_edge(int(row["src_id"]), int(row["dst_id"]), rel_type=row["rel_type"])

    # 转换为加权无向图
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        weight = 1.0
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += weight
        else:
            undirected.add_edge(u, v, weight=weight)

    # 运行Louvain算法
    partition = community_louvain.best_partition(undirected, weight="weight", resolution=1.0, random_state=42)

    # 构建结果DataFrame
    records = []
    for node_id, community_id in partition.items():
        node_data = undirected.nodes[node_id]
        records.append({
            "node_id": node_id,
            "community": community_id,
            "name": node_data.get("name", ""),
            "type": node_data.get("type", ""),
        })
    df = pd.DataFrame(records)

    # 保存社区划分结果
    output_path = data_loader.resolve_output("data/louvain/louvain_assignments.csv")
    df.to_csv(output_path, index=False)

    # 计算统计摘要
    community_sizes = df["community"].value_counts().to_dict()
    modularity = community_louvain.modularity(partition, undirected, weight="weight")

    top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
    community_insights = []
    for community_id, size in top_communities:
        subset = df[df["community"] == community_id]
        type_counts = subset["type"].value_counts().to_dict()
        disease_subset = subset[subset["type"].str.lower() == "disease"].head(5)
        key_nodes = disease_subset["name"].tolist()
        if len(key_nodes) < 5:
            remaining = subset.head(5 - len(key_nodes))
            key_nodes.extend(remaining["name"].tolist())
        community_insights.append({
            "community_id": community_id,
            "size": int(size),
            "type_distribution": type_counts,
            "sample_nodes": key_nodes,
        })

    summary = {
        "num_communities": len(community_sizes),
        "modularity": modularity,
        "largest_community_size": int(max(community_sizes.values())),
        "community_insights": community_insights,
    }

    summary_path = data_loader.resolve_output("data/louvain/louvain_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return df, summary


def _load_assignments() -> pd.DataFrame:
    """
    加载社区划分结果

    从保存的CSV文件中加载Louvain社区划分结果。

    Returns:
        pd.DataFrame: 包含node_id, community, name, type列的DataFrame

    Raises:
        ValueError: 如果CSV文件缺少必要列
    """
    path = data_loader.resolve_output("data/louvain/louvain_assignments.csv")
    df = pd.read_csv(path)
    required = {"node_id", "community", "name", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"社区划分缺少必要列: {missing}")
    return df


def _load_summary() -> Dict:
    """
    加载社区分析摘要

    从保存的JSON文件中加载社区分析统计信息。

    Returns:
        Dict: 包含社区分析统计信息的字典
    """
    path = data_loader.resolve_output("data/louvain/louvain_summary.json")
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------- Build Community-Level Graph -----------------------------
def _build_community_graph(assign: pd.DataFrame, edges: pd.DataFrame) -> Tuple[nx.Graph, Dict[int, Dict]]:
    """
    构建社区级图结构

    将原始图汇总为社区级图，其中节点表示社区，边权重表示跨社区边的数量。

    Args:
        assign: 社区划分结果DataFrame
        edges: 边数据DataFrame

    Returns:
        Tuple[nx.Graph, Dict[int, Dict]]: 社区级图和社区元数据
            - nx.Graph: 社区级图，节点为社区，边权重为跨社区边数
            - Dict[int, Dict]: 社区元数据，包含size, type_distribution, theme
    """
    # 明确类型，避免 numpy Scalar 带来的类型检查告警
    assign = assign.copy()
    if assign["community"].dtype != int:
        assign["community"] = pd.to_numeric(assign["community"], errors="coerce").fillna(-1).astype(int)
    if assign["node_id"].dtype != int:
        assign["node_id"] = pd.to_numeric(assign["node_id"], errors="coerce").fillna(-1).astype(int)
    id2comm = assign.set_index("node_id")["community"].to_dict()
    Gc: nx.Graph = nx.Graph()
    # 初始化社区节点（使用 Python int 列表避免 numpy Scalar 类型告警）
    community_ids = [int(x) for x in assign["community"].unique().tolist()]
    for cid in community_ids:
        group = assign[assign["community"] == cid]
        Gc.add_node(cid, size=int(len(group)))
    # 统计跨社区边
    weights: Dict[Tuple[int, int], int] = {}
    for _, e in edges.iterrows():
        u = int(e["src_id"]); v = int(e["dst_id"])
        cu = id2comm.get(u); cv = id2comm.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        a, b = (int(cu), int(cv)) if cu < cv else (int(cv), int(cu))
        weights[(a, b)] = weights.get((a, b), 0) + 1
    for (a, b), w in weights.items():
        Gc.add_edge(a, b, weight=w)
    # 附加类型占比/主题（主类型）
    meta: Dict[int, Dict] = {}
    for cid in community_ids:
        group = assign[assign["community"] == cid]
        counts = group["type"].value_counts().to_dict()
        main = max(counts.items(), key=lambda x: x[1])[0] if counts else "Unknown"
        meta[int(cid)] = {"size": int(len(group)), "type_distribution": counts, "theme": main}
        Gc.nodes[int(cid)]["theme"] = main
    return Gc, meta


# ----------------------------- Visualizations -----------------------------
# 节点类型颜色映射
PALETTE_TYPES = {
    "Disease": "#D62728",    # 疾病 - 红色
    "Food": "#2CA02C",       # 食物 - 绿色
    "Check": "#1F77B4",      # 检查 - 蓝色
    "Drug": "#7E57C2",       # 药品 - 紫色
    "Symptom": "#FF9800",    # 症状 - 橙色
    "Department": "#8D6E63", # 科室 - 棕色
    "Producer": "#546E7A",   # 生产商 - 灰蓝色
}

# 社区基础颜色列表
COMMUNITY_BASE_COLORS = [
    "#e53935", "#8e24aa", "#3949ab", "#1e88e5", "#00897b",
    "#7cb342", "#fdd835", "#fb8c00", "#f4511e", "#6d4c41"
]


def _lighten_color(color: str, factor: float = 0.6) -> Tuple[float, float, float, float]:
    """
    将颜色向白色混合

    Args:
        color: 十六进制颜色字符串
        factor: 混合因子，0-1之间

    Returns:
        Tuple[float, float, float, float]: RGBA颜色元组
    """
    try:
        base = np.array(mcolors.to_rgba(color))
    except ValueError:
        return (0.85, 0.85, 0.85, 1.0)
    factor = max(0.0, min(1.0, factor))
    white = np.array([1.0, 1.0, 1.0, base[3]])
    blended = base + (white - base) * factor
    return tuple(blended.tolist())


def _convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    计算点的凸包（Andrew's monotone chain算法）

    Args:
        points: 点的列表

    Returns:
        List[Tuple[float, float]]: 凸包上的点，按顺时针或逆时针顺序排列
    """
    pts = sorted(points)
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull


def _smooth_polygon(points: List[Tuple[float, float]], iterations: int = 2) -> List[Tuple[float, float]]:
    """
    使用Chaikin smoothing对多边形进行平滑处理

    Args:
        points: 多边形顶点列表
        iterations: 平滑迭代次数

    Returns:
        List[Tuple[float, float]]: 平滑后的点列表
    """
    if len(points) < 3:
        return points
    pts = points[:]
    for _ in range(iterations):
        new_pts: List[Tuple[float, float]] = []
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            q = (p1[0] * 0.75 + p2[0] * 0.25, p1[1] * 0.75 + p2[1] * 0.25)
            r = (p1[0] * 0.25 + p2[0] * 0.75, p1[1] * 0.25 + p2[1] * 0.75)
            new_pts.extend([q, r])
        pts = new_pts
    return pts


def _compute_single_community_layout(cid, node_list, edge_list, center_pos, max_size):
    """
    并行计算单个社区的局部布局

    Args:
        cid: 社区ID
        node_list: 社区内节点列表
        edge_list: 社区内边列表
        center_pos: 社区中心位置
        max_size: 最大社区大小（用于归一化）

    Returns:
        Tuple: 包含社区ID、节点位置、凸包点和节点类型的元组
    """
    group_nodes = pd.DataFrame(node_list)
    internal_edges = pd.DataFrame(edge_list)
    G_internal = nx.Graph()
    node_types_local: Dict[int, str] = {}
    for _, r in group_nodes.iterrows():
        nid = int(r.node_id)
        G_internal.add_node(nid)
        node_types_local[nid] = str(r.type)
    for _, e in internal_edges.iterrows():
        G_internal.add_edge(int(e.src_id), int(e.dst_id))
    # 局部 spring 布局，使用默认迭代次数以保持质量
    local_pos: dict
    if G_internal.number_of_nodes() > 0:
        local_pos = nx.spring_layout(G_internal, seed=int(cid), k=0.4)
    else:
        local_pos = {}
    # 缩放 & 平移
    s_norm = len(group_nodes) / max_size
    scale = 0.12 + 0.55 * s_norm
    cx, cy = center_pos[cid]
    pts: List[Tuple[float, float]] = []
    for nid, (lx, ly) in local_pos.items():
        x = cx + lx * scale
        y = cy + ly * scale
        local_pos[nid] = (x, y)  # 更新为全局位置
        pts.append((x, y))
    return cid, local_pos, pts, node_types_local




# ----------------------------- Report -----------------------------
def _write_markdown(assets: Dict[str, Path], summary: Dict, output: Path) -> Path:
    """生成详细的Louvain社区发现分析报告"""
    
    def rel(p: Path) -> str:
        return os.path.relpath(p, data_loader.REPORT_DIR)

    modularity = summary.get("modularity", 0)
    num_communities = summary.get("num_communities", 0)
    top10_insights = summary.get("community_insights", [])

    lines: List[str] = [
        f"# Louvain 社区发现分析报告",
        "",
        f"**分析日期**: {pd.Timestamp.now().strftime('%Y年%m月%d日')}  ",
        f"**数据集**: 医疗知识图谱 (社群化分析)",
        "**分析工具**: NetworkX + Python-Louvain",
        "",
        "---",
        "",
        "## 一、Louvain 社区发现算法概览",
        "",
        "Louvain 算法是一种高效的社区发现算法，用于揭示网络中的“社群”结构。在医疗知识图谱中，一个“社区”可以被理解为一组在功能上或临床上紧密相关的医疗实体（如疾病、症状、药物）的集合。",
        "该算法通过优化“模块度”（Modularity）指标来找到最佳的社区划分。模块度越高，意味着社区内部的连接远比社区之间的连接更密集，社群结构也越明显。",
        "",
        "### 1.1 核心数据",
        f"- **检测到的社区总数**: {num_communities:,}",
        f"- **网络模块度 (Modularity)**: {modularity:.4f}",
        f"- **最大社区规模**: {summary.get('largest_community_size', 0):,}",
        "",
        "#### 业务解读",
        f"**模块度为 {modularity:.4f}**，这是一个相对较高的值（通常 > 0.3 即认为网络有显著的社区结构），表明我们的医疗知识图谱并非杂乱无章，而是可以被清晰地划分为多个高内聚的“主题簇”或“功能模块”。",
        f"**检测到 {num_communities:,} 个社区**，说明医疗知识体系可以被细分为数千个不同的专业领域或疾病系统。这符合医学知识高度专业化和模块化的特点。",
        "",
        "---",
        "",
        "## 二、Top 10 社区结构分析",
        "",
        "### 2.1 Top 10 社区节点类型分布",
        "",
        f"![Top 10 社区节点类型分布]({rel(assets['type_distribution_heatmap'])})",
        "",
        "#### 业务解读",
        "上图的热力图展示了规模最大的10个社区内部的节点类型构成。每一行代表一个社区，每一列代表一种节点类型（如疾病、药物）。",
        "",
        "**结构洞察**：",
        "- **社区主题识别**：通过观察每行中颜色最深的单元格，可以快速判断一个社区的主题。例如，如果“疾病”和“症状”列的数值远高于其他列，那么这个社区很可能是一个“**疾病-症状**”主题群。",
        "- **社区功能区分**：",
        "  - **诊疗型社区**：通常包含大量的“疾病”、“症状”、“药物”和“检查”节点，形成一个完整的诊疗知识闭环。",
        "  - **预防/康复型社区**：可能以“疾病”和“食物”节点为主，侧重于生活方式和饮食干预。",
        "  - **药物专题社区**：可能由“药物”、“生产商”和相关的“疾病”构成，侧重于药理知识。",
        "",
        "**应用建议**：",
        "- **知识导航**：可以基于这些社区主题构建分层级的知识导航系统，帮助用户在特定医学领域内进行探索。",
        "- **内容聚合**：自动将同一社区内的知识点聚合起来，形成专题页面或知识卡片，提升用户体验。",
        "",
        "### 2.2 Top 10 社区簇粒子图",
        "",
        f"![Top 10 社区簇粒子图]({rel(assets['top10_clusters'])})",
        "",
        "#### 业务解读",
        "这张图将规模最大的10个社区可视化为独立的“粒子簇”，直观地展示了它们内部的结构和外部的联系。",
        "",
        "**结构洞察**：",
        "- **社区内部结构**：每个半透明区域代表一个社区。区域内节点的密集程度和连接方式反映了该社区的内部结构。例如，某些社区可能有一个或多个核心节点，而其他社区的结构则更为分散。",
        "- **社区间关联**：连接不同社区的灰色细线代表“跨界知识”。这些连接虽然稀疏，但非常重要，它们揭示了不同医学领域之间的交叉点。例如，连接“心血管社区”和“内分泌社区”的边可能代表糖尿病与心脏病的并发关系。",
        "- **节点角色**：图中不同颜色的粒子代表不同类型的节点。我们可以观察到，某些社区（如诊疗型社区）内部颜色丰富，而另一些则颜色单一（如纯粹的药物社区）。",
        "",
        "**应用建议**：",
        "- **发现交叉学科研究点**：重点分析连接两个或多个社区的“桥梁节点”，这些节点可能是多学科交叉研究的关键。",
        "- **构建智能推荐引擎**：当用户浏览一个社区的知识时，可以沿着跨社区的连接向其推荐其他相关社区的内容，实现“知识漫游”。",
        "",
        "---",
        "",
        "## 三、关键发现与洞察",
        "",
        "1. **知识的高度模块化**：高模块度（{modularity:.4f}）证明了医疗知识图谱可以被成功分解为多个高内聚、低耦合的知识模块，这为知识管理和应用提供了极大的便利。",
        "2. **社区主题的多样性**：通过对 Top 10 社区的分析，我们识别出多种主题的社区，如“疾病-症状”群、“疾病-药物”群等，揭示了知识图谱内部丰富的功能分区。",
        "3. **领域交叉点的识别**：社区间的连接虽然稀疏，但精确地标识了不同医学领域（如心血管与内分泌）之间的重要关联，为发现隐藏知识和促进多学科协作提供了线索。",
        "",
        "---",
        "",
        "## 四、应用建议",
        "",
        "- **构建分层知识图谱**：将社区作为更高层次的“概念节点”，构建一个“社区-实体”的两层知识图谱，简化宏观分析和导航。",
        "- **个性化学习路径**：根据用户的兴趣和背景，为他们规划穿越不同社区的个性化学习路径，例如从一个“症状社区”出发，深入到一个“疾病社区”，再关联到“药物社区”。",
        "- **专家知识挖掘**：识别那些专注于特定颜色（节点类型）的社区，可以帮助我们定位特定领域的专家知识集合，例如，一个几乎全由“药物”和“生产商”组成的社区，可能是一个专业的药理知识库。",
    ]

    output.write_text("\n".join(lines), encoding="utf-8")
    return output


# ----------------------------- Main -----------------------------
def generate_louvain_outputs() -> Dict[str, object]:
    """
    生成Louvain社区发现的完整输出

    执行完整的Louvain社区发现流程，包括：
    1. 计算社区划分
    2. 生成Top10社区类型分布热力图
    3. 生成Top10社区簇粒子图
    4. 生成分析报告

    Returns:
        Dict[str, object]: 包含报告路径、资源路径和元数据的字典
    """
    _configure_style()
    assign_df, summary = _compute_louvain()
    edges_df = data_loader.load_edges(["src_id", "dst_id", "rel_type"]).astype({"src_id": int, "dst_id": int})

    image_dir = _ensure_dir(data_loader.resolve_output("images/louvain"))

    assets: Dict[str, Path] = {}
    # Top10 类型分布热力图
    heat_png = image_dir / "louvain_top10_type_distribution_heatmap.png"
    heat_path = _plot_top10_type_distribution_heatmap(assign_df, heat_png)
    if heat_path:
        assets["type_distribution_heatmap"] = heat_path

    # Top10 社区真实簇粒子图
    clusters_png = image_dir / "louvain_top10_clusters.png"
    _plot_topk_particle_clusters(assign_df, edges_df, topk=10, out=clusters_png)
    assets["top10_clusters"] = clusters_png

    report_path = data_loader.REPORT_DIR / "louvain_analysis.md"
    _write_markdown(assets, summary, report_path)

    return {
        "report_markdown": report_path,
        "assets": assets,
        "meta": {"summary": summary},
    }



# ----------------------------- WordCloud -----------------------------
def _detect_chinese_font() -> str | None:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",  # macOS
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Songti.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        str(Path.home() / ".fonts" / "NotoSansCJK-Regular.ttc"),
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None

def _plot_top10_type_distribution_heatmap(assign_df: pd.DataFrame, out: Path) -> Path | None:
    """Top10 社区 × 节点类型分布热力图。"""
    # 选 Top10 社区
    size_series = assign_df.groupby("community").size().sort_values(ascending=False).head(10)
    if size_series.empty:
        return None
    top_ids = [int(c) for c in size_series.index.tolist()]
    # 统计各社区各类型数量
    sub = assign_df[assign_df["community"].isin(top_ids)].copy()
    # 仅保留需求的节点类型（去掉 科室、生产商）
    type_order = ["Disease", "Symptom", "Drug", "Food", "Check"]
    type_cn = {
        "Disease": "疾病",
        "Symptom": "症状",
        "Drug": "药品",
        "Food": "食物",
        "Check": "检查",
    }
    pivot = (sub.groupby(["community", "type"]).size().unstack(fill_value=0))
    # 只保留在数据中出现的类型，按预设顺序排列
    present_types = [t for t in type_order if t in pivot.columns]
    pivot = pivot[present_types]
    # 社区中文名：优先从 md 解析，其次自动生成
    title_map = _load_community_titles()
    missing = [cid for cid in top_ids if cid not in title_map]
    if missing:
        # 使用已存在的自动生成函数回退标题（更简洁），避免依赖已删除的生成器
        title_map.update(_auto_community_titles(assign_df, missing))
    # 重排行顺序为 Top10 顺序，并替换索引为中文
    pivot = pivot.reindex(index=top_ids)
    rename_map = {}
    for cid in top_ids:
        extra = title_map.get(cid, '')
        if extra:
            rename_map[cid] = extra
        else:
            rename_map[cid] = f"社区 {cid}"
    pivot = pivot.rename(index=rename_map)
    pivot.columns = [type_cn.get(c, c) for c in pivot.columns]
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar_kws={"label": "数量"})
    ax.set_xlabel("节点类型")
    ax.set_ylabel("社区")
    ax.set_title("Top10 社区 × 节点类型分布热力图")
    plt.tight_layout(); _ensure_dir(out.parent); fig.savefig(out, dpi=240); plt.close(fig)
    return out

def _plot_topk_particle_clusters(assign_df: pd.DataFrame, edges_df: pd.DataFrame, topk: int, out: Path) -> None:
    """绘制 TopK 社区真实簇粒子图（无采样），使用缓存和并行加速。
    """
    # 缓存文件路径
    cache_path = data_loader.resolve_output("data/louvain/layout_cache.pkl")
    
    # TopK 社区选择
    size_series = assign_df.groupby("community").size().sort_values(ascending=False).head(topk)
    top_ids = [int(c) for c in size_series.index.tolist()]
    # 过滤节点与边
    nodes_sub = assign_df[assign_df["community"].isin(top_ids)].copy()
    node_ids_set = set(nodes_sub["node_id"].tolist())
    edges_sub = edges_df[(edges_df["src_id"].isin(node_ids_set)) & (edges_df["dst_id"].isin(node_ids_set))].copy()
    # 构建社区层布局
    community_graph = nx.Graph()
    for cid in top_ids:
        community_graph.add_node(cid, size=int(size_series.loc[cid]))
    # 跨社区边统计（用于社区中心分布）
    id2comm = assign_df.set_index("node_id")["community"].to_dict()
    cross_weights: Dict[Tuple[int,int], int] = {}
    for _, e in edges_sub.iterrows():
        cu = id2comm.get(int(e["src_id"]))
        cv = id2comm.get(int(e["dst_id"]))
        if cu is None or cv is None or cu == cv:
            continue
        a,b = (cu,cv) if cu < cv else (cv,cu)
        cross_weights[(a,b)] = cross_weights.get((a,b),0)+1
    for (a,b),w in cross_weights.items():
        community_graph.add_edge(a,b, weight=w)
    center_pos = nx.spring_layout(community_graph, seed=19, k=1.4 / math.sqrt(len(community_graph)))
    max_size = int(size_series.max()) if len(size_series) else 1
    
    # 检查缓存
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        particle_positions = cached['particle_positions']
        node_types = cached['node_types']
        community_points = cached['community_points']
    else:
        # 为每个社区内部生成局部布局（并行加速，使用 ProcessPoolExecutor）
        particle_positions: Dict[int, Tuple[float,float]] = {}
        node_types: Dict[int, str] = {}
        community_points: Dict[int, List[Tuple[float,float]]] = {}
        rng = np.random.default_rng(123)
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(top_ids), 4)) as executor:
            futures = []
            for cid in top_ids:
                group_nodes = nodes_sub[nodes_sub["community"] == cid]
                internal_ids = set(group_nodes["node_id"].tolist())
                internal_edges = edges_sub[(edges_sub["src_id"].isin(internal_ids)) & (edges_sub["dst_id"].isin(internal_ids))]
                node_list = group_nodes.to_dict('records')
                edge_list = internal_edges.to_dict('records')
                futures.append(executor.submit(_compute_single_community_layout, cid, node_list, edge_list, center_pos, max_size))
            for future in concurrent.futures.as_completed(futures):
                cid, local_pos, pts, node_types_local = future.result()
                particle_positions.update(local_pos)
                node_types.update(node_types_local)
                community_points[cid] = pts
        # 保存缓存
        cached = {
            'particle_positions': particle_positions,
            'node_types': node_types,
            'community_points': community_points
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(cached, f)
    # 准备绘图
    fig, ax = plt.subplots(figsize=(12, 9))
    from matplotlib.collections import LineCollection
    # 1) 跨社区边：按(社区u,社区v)分组做轻度稀疏，先画（放在社区区域“下面”以遮挡穿越部分）
    segments_cross: List[List[Tuple[float, float]]] = []
    # 统计每对社区的边列表
    pair_edges: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for _, e in edges_sub.iterrows():
        u = int(e.src_id); v = int(e.dst_id)
        cu = id2comm.get(u); cv = id2comm.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        a, b = (int(cu), int(cv)) if cu < cv else (int(cv), int(cu))
        pair_edges.setdefault((a, b), []).append((u, v))
    # 轻度稀疏策略：每对社区最多取 min(120, max(10, ceil(0.5% * 边数))) 条
    rng = np.random.default_rng(987)
    for (a, b), ev in pair_edges.items():
        m = int(np.ceil(len(ev) * 0.005))
        keep = min(120, max(10, m))
        if len(ev) > keep:
            idxs = rng.choice(len(ev), size=keep, replace=False)
            chosen = [ev[i] for i in idxs]
        else:
            chosen = ev
        for (u, v) in chosen:
            if u in particle_positions and v in particle_positions:
                x1, y1 = particle_positions[u]; x2, y2 = particle_positions[v]
                segments_cross.append([(x1, y1), (x2, y2)])
    if segments_cross:
        lc_cross = LineCollection(segments_cross, colors="#B0BEC5", linewidths=0.25, alpha=0.18, zorder=1)
        ax.add_collection(lc_cross)
    # 社区半透明区域（凸包或椭圆）
    from matplotlib.patches import Polygon, Ellipse
    from matplotlib import patheffects as pe
    for i, cid in enumerate(top_ids):
        pts = community_points.get(cid, [])
        if not pts:
            continue
        color = COMMUNITY_BASE_COLORS[i % len(COMMUNITY_BASE_COLORS)]
        if len(pts) >= 3:
            hull = _convex_hull(pts)
            # 轻度向外膨胀（按质心）+ Chaikin 平滑
            xs = [p[0] for p in hull]; ys = [p[1] for p in hull]
            cx = sum(xs)/len(xs); cy = sum(ys)/len(ys)
            inflated = []
            for (x, y) in hull:
                dx, dy = x - cx, y - cy
                inflated.append((cx + dx * 1.05, cy + dy * 1.05))
            iters = 3 if len(hull) > 10 else 2
            smooth = _smooth_polygon(inflated, iterations=iters)
            poly = Polygon(smooth, closed=True, facecolor=color + "55", edgecolor=color, linewidth=0.9, zorder=2, joinstyle='round')
            poly.set_path_effects([
                pe.SimplePatchShadow(offset=(2, -2), alpha=0.16, rho=0.96),
                pe.PathPatchEffect(edgecolor="white", linewidth=1.1, facecolor=color + "55"),
                pe.Normal(),
            ])
            ax.add_patch(poly)
        else:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            cx = sum(xs)/len(xs); cy = sum(ys)/len(ys)
            w = (max(xs)-min(xs) + 0.05); h = (max(ys)-min(ys) + 0.05)
            ell = Ellipse((cx, cy), width=w*1.1, height=h*1.1, facecolor=color + "55", edgecolor=color, linewidth=0.9, zorder=2)
            ell.set_path_effects([
                pe.SimplePatchShadow(offset=(2, -2), alpha=0.16, rho=0.96),
                pe.PathPatchEffect(edgecolor="white", linewidth=1.1, facecolor=color + "55"),
                pe.Normal(),
            ])
            ax.add_patch(ell)
    # 2) 社区内部边：按社区分别着色（使用社区主色半透明），放在半透明区域之上
    segments_intra_by_comm: Dict[int, List[List[Tuple[float, float]]]] = {cid: [] for cid in top_ids}
    for _, e in edges_sub.iterrows():
        u = int(e.src_id); v = int(e.dst_id)
        cu = id2comm.get(u); cv = id2comm.get(v)
        if cu is None or cv is None or cu != cv:
            continue
        if u in particle_positions and v in particle_positions:
            x1, y1 = particle_positions[u]; x2, y2 = particle_positions[v]
            segments_intra_by_comm[int(cu)].append([(x1, y1), (x2, y2)])
    for i, cid in enumerate(top_ids):
        segs = segments_intra_by_comm.get(cid, [])
        if not segs:
            continue
        # 将所有社区内部边统一使用淡灰色（更弱的视觉优先级），而非按社区色彩着色
        intra_color = "#cccccc"
        lc_intra = LineCollection(segs, colors=intra_color, linewidths=0.35, alpha=0.45, zorder=3)
        ax.add_collection(lc_intra)
    # 不在社区内部标注编号或规模
    # 绘制粒子（最高图层）
    xs_all = [particle_positions[n][0] for n in particle_positions]
    ys_all = [particle_positions[n][1] for n in particle_positions]
    colors_all = [PALETTE_TYPES.get(node_types[n], "#B0BEC5") for n in particle_positions]
    ax.scatter(xs_all, ys_all, c=colors_all, s=5, alpha=0.85, linewidths=0, zorder=4)
    # 图例：社区名称
    from matplotlib.patches import Patch
    # 3) 图例标题：优先 md 中的中文标题，缺失则自动生成
    title_map = _load_community_titles()
    missing = [cid for cid in top_ids if cid not in title_map]
    if missing:
        auto_map = _auto_community_titles(assign_df, missing)
        title_map.update(auto_map)
    legend_handles = []
    legend_labels = []
    for i, cid in enumerate(top_ids):
        color = COMMUNITY_BASE_COLORS[i % len(COMMUNITY_BASE_COLORS)]
        label = f"社区 {cid}"
        extra = title_map.get(cid)
        if extra:
            label = extra
        legend_handles.append(Patch(facecolor=color+"80", edgecolor=color))
        legend_labels.append(label)
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper left", fontsize=8, frameon=False)
    ax.set_title(f"Top{topk} 社区簇分布图")
    ax.set_axis_off(); plt.tight_layout(); _ensure_dir(out.parent); fig.savefig(out, dpi=260); plt.close(fig)

if __name__ == "__main__":
    generate_louvain_outputs()

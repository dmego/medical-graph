"""医疗知识图谱 HITS (Hub/Authority) 分析模块

生成 HITS 算法的可视化和业务洞察报告。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, cast

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib.ticker import FuncFormatter

from . import data_loader
from .set_chinese_plot import configure_chinese_plot_style


# ----------------------------- 样式和工具函数 -----------------------------
def _configure_style() -> None:
    """配置matplotlib中文显示样式"""
    configure_chinese_plot_style()


def _ensure_dir(path: Path) -> Path:
    """确保目录存在，如果不存在则创建"""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------------- 数据加载函数 -----------------------------
def _compute_hits_scores() -> pd.DataFrame:
    """计算HITS算法的hub和authority得分"""
    nodes_df = data_loader.load_nodes()
    edges_df = data_loader.load_edges()
    G = nx.from_pandas_edgelist(edges_df, source='src_id', target='dst_id', create_using=nx.DiGraph)
    hubs, authorities = nx.hits(G)
    df = nodes_df.copy()
    df['hub_score'] = df['node_id'].map(hubs).fillna(0)
    df['authority_score'] = df['node_id'].map(authorities).fillna(0)
    output_path = data_loader.resolve_output("data/hits/hits_scores.csv")
    df.to_csv(output_path, index=False)
    return df


def _load_pagerank_scores() -> pd.DataFrame | None:
    """加载PageRank得分数据"""
    try:
        path = data_loader.resolve_output("data/pagerank/pagerank_scores.csv")
        pr = pd.read_csv(path, usecols=["node_id", "pagerank"]).set_index("node_id")
        return pr
    except FileNotFoundError:
        return None


# ----------------------------- 数据处理函数 -----------------------------
def _top(df: pd.DataFrame, col: str, top_n: int, type_filter: str | None = None) -> pd.DataFrame:
    """获取指定列得分最高的top_n个节点"""
    sub = df if type_filter is None else df[df["type"] == type_filter]
    if sub.empty:
        return sub
    return sub.nlargest(top_n, col).reset_index(drop=True)


def _compute_divergence(hits: pd.DataFrame, pagerank: pd.DataFrame | None) -> pd.DataFrame:
    """计算HITS得分与PageRank得分的差异"""
    if pagerank is None:
        return pd.DataFrame()
    merged = hits.set_index("node_id").join(pagerank, on="node_id")
    if "pagerank" not in merged.columns:
        return pd.DataFrame()

    # 计算z-score标准化
    for col in ("hub_score", "authority_score", "pagerank"):
        mean = float(merged[col].mean())
        std = float(merged[col].std()) or 1.0
        merged[col + "_z"] = (merged[col] - mean) / std

    merged["hub_vs_pr_diff"] = merged["hub_score_z"] - merged["pagerank_z"]
    merged["auth_vs_pr_diff"] = merged["authority_score_z"] - merged["pagerank_z"]
    return merged.reset_index(drop=False)


# ----------------------------- 可视化函数 -----------------------------
def _bar_top(df: pd.DataFrame, col: str, out: Path, title: str) -> pd.DataFrame:
    """绘制得分最高的节点条形图"""
    top = df.head(10).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = "Reds" if col == "hub_score" else "Blues"
    sns.barplot(data=top, y="name", x=col, ax=ax, hue="name", palette=colors, legend=False)
    ax.set_xlabel(f"{col} 得分")
    ax.set_ylabel("节点")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2e}"))
    plt.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return top


def _network_top_hub_diseases(hits: pd.DataFrame, edges: pd.DataFrame, out_png: Path, top_n: int = 8) -> Dict[str, int]:
    """绘制Top Hub疾病的下游网络图"""
    diseases = _top(hits, "hub_score", top_n, type_filter="Disease")
    if diseases.empty:
        return {"diseases": 0, "downstream": 0, "edges": 0}

    disease_ids = diseases["node_id"].tolist()
    nodes_meta = hits.set_index("node_id")

    # 筛选下游关系
    sub_edges = edges[edges["src_id"].isin(disease_ids)]
    sub_edges = sub_edges[sub_edges["rel_type"].isin({"common_drug", "recommand_drug", "recommand_eat", "need_check"})]

    if sub_edges.empty:
        return {"diseases": len(disease_ids), "downstream": 0, "edges": 0}

    # 构建网络图
    G = nx.DiGraph()
    palette = {
        "Disease": "#D62728",
        "Food": "#2CA02C",
        "Check": "#1F77B4",
        "Drug": "#7E57C2",
        "Symptom": "#FF9800",
        "Department": "#8D6E63",
        "Producer": "#546E7A",
    }

    # 添加疾病节点
    for _, row in diseases.iterrows():
        G.add_node(str(row.node_id), label=str(row["name"]), type="Disease", hub=float(row.hub_score))

    # 添加下游节点和边
    for _, e in sub_edges.iterrows():
        src = str(e.src_id)
        dst_id = int(e.dst_id)
        if dst_id not in nodes_meta.index:
            continue
        node_row = nodes_meta.loc[dst_id]
        node_type = str(node_row.get("type", ""))
        if node_type not in {"Food", "Check", "Drug"}:
            continue
        hub_raw = node_row.get("hub_score", 0.0)
        if isinstance(hub_raw, pd.Series):
            hub_raw = hub_raw.iloc[0] if not hub_raw.empty else 0.0
        hub_f = float(hub_raw) if isinstance(hub_raw, (int, float)) else 0.0
        G.add_node(str(dst_id), label=str(node_row.get("name", "")), type=node_type, hub=hub_f)
        G.add_edge(src, str(dst_id), rel_type=str(e.rel_type))

    # 绘制静态网络图
    UNIFORM_SIZE = 16
    pos = nx.spring_layout(G, seed=7, k=0.6)
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制不同类型的节点
    for node_type, color in palette.items():
        nodes_t = [n for n, d in G.nodes(data=True) if d.get("type") == node_type]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_t, node_color=color,
                               node_size=UNIFORM_SIZE * 14, alpha=0.85, ax=ax, label=node_type)

    # 绘制边
    nx.draw_networkx_edges(G, pos, alpha=0.55, edge_color="#9E9E9E", ax=ax, arrows=False)

    # 添加节点标签
    for n, d in G.nodes(data=True):
        x, y = pos[n]
        label = str(d.get("label", ""))
        ax.text(x + 0.02, y, label, fontsize=8, ha="left", va="center")

    ax.set_axis_off()
    ax.set_title("Top Hub 疾病的下游多类型连接图")
    ax.legend(loc="lower left")
    plt.tight_layout()
    _ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=240)
    plt.close(fig)

    return {
        "diseases": len(disease_ids),
        "downstream": sum(1 for n, d in G.nodes(data=True) if d.get('type') != 'Disease'),
        "edges": G.number_of_edges()
    }


def _authority_bar(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """绘制Top Authority节点的条形图"""
    top = _top(df, "authority_score", 10)[["name", "authority_score", "type"]].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {
        "Food": "#2CA02C",
        "Check": "#1F77B4",
        "Symptom": "#FF9800",
        "Drug": "#7E57C2",
        "Disease": "#D62728",
        "Department": "#8D6E63",
        "Producer": "#546E7A",
    }
    sns.barplot(data=top, y="name", x="authority_score", hue="type", ax=ax, palette=palette)
    ax.set_xlabel("Authority 得分")
    ax.set_ylabel("节点")
    ax.set_title("Top 权威节点 (Authority)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2e}"))
    ax.legend(title="类型", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return top


def _scatter_compare(div: pd.DataFrame, col: str, out: Path) -> None:
    """绘制PageRank与指定列的对比散点图"""
    if div.empty or col not in {"hub_score", "authority_score"}:
        return

    fig, ax = plt.subplots(figsize=(8.2, 6))
    sns.scatterplot(data=div, x="pagerank", y=col, hue="type", alpha=0.4, s=20, ax=ax)
    ax.set_xlabel("PageRank 得分")
    ax.set_ylabel(f"{col} 得分")
    ax.set_title(f"PageRank vs {col} 对比")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2e}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2e}"))

    # 标注Top5节点名称
    try:
        top5 = div.nlargest(5, col)
        for _, r in top5.iterrows():
            name = str(r.get("name", ""))
            ax.text(r["pagerank"], r[col], name, fontsize=8, color="#333333")
    except Exception:
        pass

    plt.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


# ----------------------------- Markdown Report -----------------------------
def _write_markdown(assets: Dict[str, Path | None], meta: Dict[str, object], output: Path) -> Path:
    """生成详细的HITS分析报告"""
    hits_df = cast(pd.DataFrame, meta["hits_df"])
    top_hub_diseases = cast(pd.DataFrame, meta["top_hub_diseases"])
    top_authorities = cast(pd.DataFrame, meta["top_authorities"])
    network_stats = cast(Dict[str, int], meta["network_stats"])
    divergence_df = cast(pd.DataFrame, meta.get("divergence_df", pd.DataFrame()))

    def rel(p: Path | None) -> str:
        if not isinstance(p, Path):
            return ""
        return os.path.relpath(p, data_loader.REPORT_DIR)

    # 提取核心数据
    avg_hub = hits_df['hub_score'].mean()
    avg_auth = hits_df['authority_score'].mean()
    top_hub_disease_list = top_hub_diseases['name'].tolist()
    top_authority_list = top_authorities[['name', 'type']].to_dict('records')

    lines: List[str] = [
        f"# 医疗知识图谱 HITS (Hub/Authority) 分析报告",
        "",
        f"**分析日期**: {pd.Timestamp.now().strftime('%Y年%m月%d日')}  ",
        f"**数据集**: 医疗知识图谱 (节点: {len(hits_df):,})",
        "**分析工具**: NetworkX + Python",
        "",
        "---",
        "",
        "## 一、HITS 算法概览",
        "",
        "HITS (Hyperlink-Induced Topic Search) 是一种衡量节点在网络中重要性的算法，它将节点的重要性分为两个维度：",
        "- **Hub (枢纽) Score**: 指向大量高质量 Authority 页面的节点，其 Hub 分数会更高。在医疗图谱中，高 Hub 节点通常是“信息分发中心”，如复杂疾病，它们关联多种药物、检查和症状。",
        "- **Authority (权威) Score**: 被大量高质量 Hub 页面指向的节点，其 Authority 分数会更高。在医疗图谱中，高 Authority 节点通常是“公认的优质资源”，如特效药、关键检查手段或典型症状。",
        "",
        "### 1.1 核心数据",
        f"- **节点总数**: {len(hits_df):,}",
        f"- **平均 Hub 分数**: {avg_hub:.2e}",
        f"- **平均 Authority 分数**: {avg_auth:.2e}",
        "",
        "---",
        "",
        "## 二、Hub (枢纽) 分析",
        "",
        "### 2.1 Top 10 枢纽疾病 (Hub Score)",
        "",
        f"![Top 10 枢纽疾病]({rel(assets.get('bar_hub_disease'))})",
        "",
        "#### 业务解读",
        "Hub 分数衡量一个节点指向其他“权威”节点的能力。在医疗领域，高 Hub 分数的疾病通常是**复杂性高、涉及多系统、需要多种手段干预**的疾病。",
        "",
        f"**数据观察**：",
        f"上图展示了 Hub 分数最高的10个疾病，例如 **{top_hub_disease_list[0]}**、**{top_hub_disease_list[1]}** 和 **{top_hub_disease_list[2]}**。这些疾病的共同特点是：",
        "- **诊疗路径复杂**：它们通常不是单一病因或单一疗法能解决的，需要关联多种药物、检查、饮食建议甚至并发症管理。",
        "- **信息枢纽价值高**：在构建临床决策支持系统或患者教育内容时，这些疾病是天然的“入口”，可以引导用户探索相关的多方面知识。",
        "",
        "**应用建议**：",
        "- **智能导诊入口**：可将这些高 Hub 疾病作为智能导诊的优先推荐入口，因为它们覆盖的下游知识范围更广。",
        "- **内容专题策划**：围绕这些疾病制作深度科普内容（如“一文读懂XXX”），能够有效组织和串联大量相关医疗知识。",
        "",
        "### 2.2 Top Hub 疾病的下游网络",
        "",
        f"![Top Hub 疾病的下游网络]({rel(assets.get('network_hub_static'))})",
        "",
        "#### 业务解读",
        f"为了进一步理解 Top Hub 疾病的“枢纽”特性，我们构建了其下游关联网络。该网络包含 **{network_stats.get('diseases', 0)}** 个核心疾病节点，连接了 **{network_stats.get('downstream', 0)}** 个下游资源节点（药物、食品、检查），共计 **{network_stats.get('edges', 0)}** 条连接。",
        "",
        "**结构洞察**：",
        "- **资源多样性**：从图中可以看出，一个高 Hub 疾病（如中心位置的红色节点）会同时连接多种颜色（代表不同类型）的下游节点，这直观地展示了其诊疗的“多模态”特性。",
        "- **连接密度**：某些疾病周围的下游节点非常密集，表明其治疗和管理方案标准化程度高、选择多。而连接稀疏的区域可能代表这是一个研究较少或治疗方案有限的领域。",
        "",
        "**应用建议**：",
        "- **构建“疾病画像”**：该网络结构是构建“疾病知识卡片”或“疾病画像”的绝佳数据基础，能全面展示与某一疾病相关的所有核心信息。",
        "- **发现知识缺口**：如果一个常见复杂疾病在图中的下游连接（如“推荐药物”或“必要检查”）非常稀疏，这可能指示了知识库中存在内容缺口，需要优先补充。",
        "",
        "---",
        "",
        "## 三、Authority (权威) 分析",
        "",
        "### 3.1 Top 10 权威节点 (Authority Score)",
        "",
        f"![Top 10 权威节点]({rel(assets.get('bar_authority'))})",
        "",
        "#### 业务解读",
        "Authority 分数衡量一个节点被多少“枢纽”节点指向。在医疗领域，高 Authority 分数的节点通常是**被广泛引用和推荐的核心资源**。",
        "",
        "**数据观察**：",
        f"上图展示了 Authority 分数最高的10个节点。我们发现，这些节点类型多样，包括 **{top_authority_list[0]['type']}** (如“{top_authority_list[0]['name']}”) 和 **{top_authority_list[1]['type']}** (如“{top_authority_list[1]['name']}”)。",
        "- **通用性与关键性**：高 Authority 节点往往是多种疾病诊疗路径上的“共同终点”。例如，某个检查项目被多种复杂疾病作为关键诊断依据，或者某种药物是多种疾病的常规治疗选择。",
        "- **“黄金标准”资源**：这些节点可以被视为医疗知识库中的“黄金标准”或“高价值”资源，因为它们被最多的上游复杂场景所需要。",
        "",
        "**应用建议**：",
        "- **知识库质量提升**：应优先确保这些高 Authority 节点的知识描述最准确、最详尽，因为它们被引用的频率最高。",
        "- **搜索与推荐优化**：在搜索引擎中，可以为这些节点赋予更高权重。在推荐系统中，它们可以作为“明星资源”优先展示。",
        "",
        "---",
        "",
        "## 四、HITS 与 PageRank 对比分析",
        "",
        f"![PageRank vs Authority 对比]({rel(assets.get('compare_pagerank_authority'))})",
        "",
        "#### 业务解读",
        "PageRank 衡量全局的重要性（被越多、越重要的节点指向，得分越高），而 Authority 衡量作为“权威资源”被“枢纽”指向的程度。对比两者可以发现不同类型的核心节点。",
        "",
        "**象限分析**：",
        "- **高 Authority / 高 PageRank (右上)**：图谱中的“超级明星”。它们既是全局的核心（PageRank高），也是特定领域的权威资源（Authority高）。这些是知识库中最关键的节点。",
        "- **高 Authority / 低 PageRank (左上)**：专业的“领域权威”。它们可能不是全局热点，但在特定复杂场景下是不可或缺的关键资源。例如，一种罕见病的特效药或特定手术的关键检查。",
        "- **低 Authority / 高 PageRank (右下)**：重要的“连接者”或“基础概念”。它们在全图中连接广泛，但并非被复杂疾病集中引用的“最终答案”。",
        "",
        "**应用建议**：",
        "- **差异化运营**：“超级明星”节点适合大众科普和广泛推荐。“领域权威”节点适合用于专家系统和垂直领域的深度内容建设。",
        "",
        "---",
        "",
        "## 五、关键发现与洞察",
        "",
        "1. **识别核心疾病**：HITS 的 Hub 分数成功识别了一批诊疗路径复杂、涉及多学科的“枢纽疾病”，它们是知识图谱中的关键入口。",
        "2. **发现权威资源**：Authority 分数有效挖掘出被多种复杂疾病共同依赖的“黄金标准”资源，如关键检查和通用药物。",
        "3. **揭示网络结构**：Hub-Authority 结构清晰地展示了医疗知识“从疾病到资源”的放射状关联模式，与 `graph_analysis` 中发现的星型结构一致。",
        "4. **补充全局视角**：与 PageRank 的对比分析，为我们提供了更丰富的节点角色定义，区分了“全局明星”与“领域专家”，有助于实施更精细化的运营策略。",
        "",
        "---",
        "",
        "## 六、应用建议",
        "",
        "- **智能导诊优化**：利用 Top Hub 疾病作为导诊的起点，并根据其下游网络构建多轮问答与推荐逻辑。",
        "- **内容质量与优先级**：优先审核和丰富 Top Authority 节点的内容，确保这些最高频被引用的知识准确无误。",
        "- **个性化推荐**：结合 Hub 和 Authority 分析，可以为用户构建“从症状/疾病（Hub）到推荐方案（Authority）”的个性化知识路径。",
        "- **知识图谱补全**：分析低 Hub 或低 Authority 的区域，识别知识覆盖的薄弱环节，并指导数据补充和知识挖掘的方向。",
    ]

    output.write_text("\n".join(lines), encoding="utf-8")
    return output


# ----------------------------- Main Generation -----------------------------
def generate_hits_outputs() -> Dict[str, object]:
    _configure_style()
    hits_df = _compute_hits_scores()
    edges_df = data_loader.load_edges(["src_id", "dst_id", "rel_type"])
    pagerank_df = _load_pagerank_scores()

    image_dir = _ensure_dir(data_loader.resolve_output("images/hits"))
    # 不再生成交互 HTML

    # Top Hub 疾病
    top_hub_diseases = _top(hits_df, "hub_score", 10, type_filter="Disease")
    bar_hub_disease = image_dir / "hits_top_hub_diseases.png"
    _bar_top(top_hub_diseases, "hub_score", bar_hub_disease, "Top 疾病 Hub 得分")

    # 下游网络
    network_png = image_dir / "hits_hub_network.png"
    network_stats = _network_top_hub_diseases(hits_df, edges_df, network_png)

    # Top Authority 节点
    top_authorities = _top(hits_df, "authority_score", 10)
    bar_authority = image_dir / "hits_top_authorities.png"
    _authority_bar(top_authorities, bar_authority)

    # 删除疾病->权威热力图，不再生成

    # 与 PageRank 的对照分析（Authority）
    divergence_df = _compute_divergence(hits_df, pagerank_df)
    compare_auth = image_dir / "hits_compare_pagerank_authority.png"
    _scatter_compare(divergence_df, "authority_score", compare_auth)

    assets: Dict[str, Path | None] = {
        "bar_hub_disease": bar_hub_disease,
        "network_hub_static": network_png,
        "bar_authority": bar_authority,
        "compare_pagerank_authority": compare_auth if compare_auth.exists() else None,
    }

    meta: Dict[str, object] = {
        "hits_df": hits_df,
        "top_hub_diseases": top_hub_diseases,
        "top_authorities": top_authorities,
        "network_stats": network_stats,
    }

    report_path = data_loader.REPORT_DIR / "hits_analysis.md"
    _write_markdown(assets, meta, report_path)

    return {
        "report_markdown": report_path,
        "assets": assets,
        "meta": meta,
    }


if __name__ == "__main__":
    generate_hits_outputs()

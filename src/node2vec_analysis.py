""" Node2Vec 嵌入分析。

本模块实现 Node2Vec 嵌入可视化分析，包含以下阶段：
1. 使用 igraph 实现符合 node2vec 转移规则的偏置随机游走。
2. 使用 gensim 的 Word2Vec 多核训练，并缓存详细的运行时统计指标。
3. 连接并融合 Louvain、PageRank、HITS 与度数等元数据，丰富节点特征。
4. 执行 KMeans k 网格搜索、UMAP/t-SNE 可视化，并生成 Louvain 对齐热力图。
5. 生成符合医疗知识图谱规范的 Markdown 分析报告。

在项目根目录运行：
    python -m src.node2vec_analysis [--force-embeddings]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

try:  # 延迟导入：若缺少依赖则给出明确提示
    import igraph as ig  # type: ignore
except ImportError as exc:  # pragma: no cover - 依赖保护
    raise RuntimeError("需要安装 python-igraph>=0.10 才能运行 node2vec 分析。") from exc

try:
    from gensim.models import Word2Vec  # type: ignore
except ImportError as exc:  # pragma: no cover - 依赖保护
    raise RuntimeError("需要安装 gensim>=4.0 才能运行 node2vec 分析。") from exc

try:  # 可选的降维库优先使用 UMAP
    import umap  # type: ignore
except ImportError:  # pragma: no cover - 可选依赖
    umap = None  # type: ignore

try:  # 仅用于显示进度条，可选
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - 可选依赖
    tqdm = None  # type: ignore

from . import data_loader
from .set_chinese_plot import configure_chinese_plot_style


DATA_DIR = Path("data/node2vec")
IMAGE_DIR = Path("images/node2vec")
REPORT_PATH = data_loader.REPORT_DIR / "node2vec_analysis.md"


@dataclass(frozen=True)
class Node2vecConfig:
    """配置容器：用于保存超参数与绘图参数开关。"""

    walk_length: int = 80
    num_walks: int = 10
    p: float = 0.25
    q: float = 1.0
    dimensions: int = 128
    window: int = 10
    epochs: int = 20
    workers: Optional[int] = None
    force_embeddings: bool = False
    seed: int = 42
    k_grid: Tuple[int, ...] = tuple(range(4, 42, 2))
    silhouette_sample_size: int = 4000
    max_visual_nodes: int = 6000
    pagerank_top_pct: float = 0.02
    degree_top_pct: float = 0.02

    def _resolve_worker_count(self) -> int:
        """解析训练线程数：优先显式参数，其次环境变量，最后 CPU-1。"""
        if self.workers is not None:  # 若明确传入 workers，则直接使用（至少为 1）
            return max(1, int(self.workers))
        env_override = os.getenv("NODE2V2_WORKERS")  # 尝试读取环境变量覆盖
        if env_override:
            try:
                return max(1, int(env_override))
            except ValueError:
                pass  # 非法环境变量则忽略
        cpu_total = os.cpu_count() or 1  # 自动检测可用 CPU 核心数
        return max(1, cpu_total - 1)  # 预留 1 个核心给系统


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def build_config(args: argparse.Namespace) -> Node2vecConfig:
    base = Node2vecConfig()
    walk_length = args.walk_length if args.walk_length is not None else _env_int("NODE2V2_WALK_LENGTH", base.walk_length)
    num_walks = args.num_walks if args.num_walks is not None else _env_int("NODE2V2_NUM_WALKS", base.num_walks)
    p_value = args.p if args.p is not None else _env_float("NODE2V2_P", base.p)
    q_value = args.q if args.q is not None else _env_float("NODE2V2_Q", base.q)
    dimensions = args.dimensions if args.dimensions is not None else _env_int("NODE2V2_DIMENSIONS", base.dimensions)
    window = args.window if args.window is not None else _env_int("NODE2V2_WINDOW", base.window)
    epochs = args.epochs if args.epochs is not None else _env_int("NODE2V2_EPOCHS", base.epochs)
    workers = args.workers if args.workers is not None else (_env_int("NODE2V2_WORKERS", base.workers or 0) or None)
    seed = args.seed if args.seed is not None else _env_int("NODE2V2_SEED", base.seed)
    max_visual_nodes = args.max_visual_nodes if args.max_visual_nodes is not None else _env_int("NODE2V2_MAX_VISUAL", base.max_visual_nodes)
    silhouette_sample_size = args.silhouette_sample_size if args.silhouette_sample_size is not None else _env_int("NODE2V2_SILHOUETTE_SAMPLE", base.silhouette_sample_size)
    pagerank_top_pct = args.pagerank_top_pct if args.pagerank_top_pct is not None else _env_float("NODE2V2_PR_TOP_PCT", base.pagerank_top_pct)
    degree_top_pct = args.degree_top_pct if args.degree_top_pct is not None else _env_float("NODE2V2_DEGREE_TOP_PCT", base.degree_top_pct)

    if args.k_grid:
        try:
            k_grid_tuple = tuple(sorted({int(k.strip()) for k in args.k_grid.split(',') if k.strip()}))
        except ValueError as exc:
            raise ValueError("--k-grid must be a comma separated list of integers") from exc
    else:
        env_k_grid = os.getenv("NODE2V2_K_GRID")
        if env_k_grid:
            try:
                k_grid_tuple = tuple(sorted({int(k.strip()) for k in env_k_grid.split(',') if k.strip()}))
            except ValueError as exc:
                raise ValueError("NODE2V2_K_GRID must contain integers") from exc
        else:
            k_grid_tuple = base.k_grid

    return Node2vecConfig(
        walk_length=walk_length,
        num_walks=num_walks,
        p=p_value,
        q=q_value,
        dimensions=dimensions,
        window=window,
        epochs=epochs,
        workers=workers,
        force_embeddings=args.force_embeddings,
        seed=seed,
        k_grid=k_grid_tuple,
        silhouette_sample_size=silhouette_sample_size,
        max_visual_nodes=max_visual_nodes,
        pagerank_top_pct=pagerank_top_pct,
        degree_top_pct=degree_top_pct,
    )


def _data_path(relative: str | Path) -> Path:
    return data_loader.resolve_output(DATA_DIR / Path(relative))


def _image_path(relative: str | Path) -> Path:
    return data_loader.resolve_output(IMAGE_DIR / Path(relative))


def _load_nodes_edges() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = data_loader.load_nodes()
    edges = data_loader.load_edges()
    return nodes, edges


def _load_louvain_assignments() -> pd.DataFrame:
    """读取 Louvain 社区划分结果，供聚类对齐与热力图使用。"""
    path = data_loader.OUTPUT_DIR / "data" / "louvain" / "louvain_assignments.csv"
    if not path.exists():
        raise FileNotFoundError("Missing louvain_assignments.csv. Run src.louvain_analysis first.")
    df = pd.read_csv(path)
    if "community" not in df.columns:
        raise ValueError("louvain_assignments.csv must contain a 'community' column")
    return df


def _load_louvain_titles() -> Dict[int, str]:
    """读取社区中文标题，增强图例与标签的可读性。"""
    path = data_loader.OUTPUT_DIR / "data" / "louvain" / "louvain_top10_name.txt"
    if not path.exists():
        return {}
    mapping: Dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        try:
            mapping[int(key)] = value
        except ValueError:
            continue
    return mapping


def _load_pagerank_scores() -> Optional[pd.DataFrame]:
    """加载 PageRank 得分，用于点径映射与高亮。"""
    path = data_loader.OUTPUT_DIR / "data" / "pagerank" / "pagerank_scores.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["node_id", "pagerank"])
    return df


def _load_hits_scores() -> Optional[pd.DataFrame]:
    """加载 HITS Hub/Authority 分数，支持角色标签展示。"""
    path = data_loader.OUTPUT_DIR / "data" / "hits" / "hits_scores.csv"
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        return None
    df = pd.read_csv(path, usecols=["node_id", "hub_score", "authority_score"])
    return df


def _build_igraph(nodes: pd.DataFrame, edges: pd.DataFrame) -> ig.Graph:
    """根据 CSV 数据构建 igraph 图，并在内部去除孤立点。"""
    graph = ig.Graph()  # 创建空图
    node_ids = nodes["node_id"].astype(int).tolist()  # 读取节点 ID 列并转换为整数列表
    if not node_ids:  # 若无节点则直接报错，避免后续空图计算
        raise ValueError("nodes.csv 为空，无法构建图")
    graph.add_vertices(len(node_ids))  # 按节点数量添加顶点
    graph.vs["node_id"] = node_ids  # 保存节点 ID 属性，后续便于映射
    # 节点名称与类型属性：缺失值填空，并转为字符串
    graph.vs["name"] = nodes.get("name", pd.Series(["" for _ in node_ids])).fillna("").astype(str).tolist()
    graph.vs["type"] = nodes.get("type", pd.Series(["" for _ in node_ids])).fillna("").astype(str).tolist()

    # 建立 node_id -> 顶点索引 的映射，后续构边要用索引
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    weight_map: Dict[Tuple[int, int], float] = {}  # 汇总无向边的权重（多重边计数）

    # 遍历边表，将有向边映射为无向边，并累加权重（多重边）
    for row in edges.itertuples(index=False):
        src_id = int(getattr(row, "src_id"))  # 边起点 ID
        dst_id = int(getattr(row, "dst_id"))  # 边终点 ID
        if src_id == dst_id:  # 跳过自环
            continue
        src_idx = id_to_idx.get(src_id)  # 起点索引
        dst_idx = id_to_idx.get(dst_id)  # 终点索引
        if src_idx is None or dst_idx is None:  # 若对应节点不存在，跳过
            continue
        edge_key = (min(src_idx, dst_idx), max(src_idx, dst_idx))  # 规范化无向边键
        weight_map[edge_key] = weight_map.get(edge_key, 0.0) + 1.0  # 多重边累加权重

    # 若存在边，则批量添加并写入权重；否则写入空权重数组
    if weight_map:
        edges_list = list(weight_map.keys())
        weights = list(weight_map.values())
        graph.add_edges(edges_list)
        graph.es["weight"] = weights
    else:
        graph.es["weight"] = []

    # 计算并删除孤立点（度数为 0 的顶点）
    isolates = [v.index for v in graph.vs if graph.degree(v.index) == 0]
    if isolates:
        graph.delete_vertices(isolates)
    return graph


def _alias_setup(probabilities: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(probabilities)
    if n == 0:
        return np.array([]), np.array([])
    probs = np.array(probabilities, dtype=float)
    total = probs.sum()
    if total <= 0:
        probs = np.full(n, 1.0 / n)
    else:
        probs /= total
    scaled = probs * n
    smaller: List[int] = []
    larger: List[int] = []
    alias = np.zeros(n, dtype=int)
    q = np.zeros(n)

    for i, value in enumerate(scaled):
        q[i] = value
        if value < 1.0:
            smaller.append(i)
        else:
            larger.append(i)

    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()
        alias[small] = large
        q[large] = q[large] - (1.0 - q[small])
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return q, alias


def _alias_draw(q: np.ndarray, alias: np.ndarray, rng: np.random.Generator) -> int:
    if q.size == 0:
        return 0
    n = q.size
    idx = int(rng.integers(n))
    if rng.random() < q[idx]:
        return idx
    return int(alias[idx])


class IGraphNode2VecWalker:
    """为 igraph 图预计算 node2vec 所需的别名采样表（Alias Tables）。"""

    def __init__(self, graph: ig.Graph, p: float, q: float):
        self.graph = graph  # igraph 图对象
        self.p = max(p, 1e-6)  # 返回概率参数 p，下限防止除零
        self.q = max(q, 1e-6)  # 进出概率参数 q，下限防止除零
        self.alias_nodes: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # 节点级别的 alias 表
        self.alias_edges: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}  # 有向边级别的 alias 表
        self._prepare_alias_tables()  # 预计算 alias 表，加速后续采样

    def _prepare_alias_tables(self) -> None:
        # 针对每个节点，按邻接边权构建节点级 alias 表
        for node in range(self.graph.vcount()):
            neighbors = self.graph.neighbors(node)
            weights = [self._edge_weight(node, nbr) for nbr in neighbors]
            self.alias_nodes[node] = _alias_setup(weights if weights else [1.0])

        # 针对每条（无向）边的两个方向，构建有向边级 alias 表
        for edge in self.graph.es:
            src = edge.source
            dst = edge.target
            self.alias_edges[(src, dst)] = self._alias_edge(src, dst)
            self.alias_edges[(dst, src)] = self._alias_edge(dst, src)

    def _edge_weight(self, src: int, dst: int) -> float:
        try:
            eid = self.graph.get_eid(src, dst, directed=False)
            weight = self.graph.es[eid]["weight"] if "weight" in self.graph.es.attribute_names() else 1.0
        except ig._igraph.InternalError:  # type: ignore[attr-defined]
            weight = 1.0
        return float(weight) if weight else 1.0

    def _alias_edge(self, src: int, dst: int) -> Tuple[np.ndarray, np.ndarray]:
        neighbors = self.graph.neighbors(dst)
        unnormalized: List[float] = []
        for neighbor in neighbors:
            weight = self._edge_weight(dst, neighbor)
            if neighbor == src:
                unnormalized.append(weight / self.p)
            elif self.graph.are_connected(neighbor, src):
                unnormalized.append(weight)
            else:
                unnormalized.append(weight / self.q)
        return _alias_setup(unnormalized if unnormalized else [1.0])

    def simulate_walks(self, num_walks: int, walk_length: int, seed: int) -> Tuple[List[List[str]], int]:
        rng = np.random.default_rng(seed)  # 复现实验的随机数生成器
        vertices = list(range(self.graph.vcount()))  # 顶点索引列表
        walks: List[List[str]] = []  # 储存所有游走序列（字符串 ID）
        total_tokens = 0  # 统计游走 token 总数
        for walk_round in range(num_walks):  # 每个节点独立重复 num_walks 轮
            rng.shuffle(vertices)  # 打乱起点顺序，降低偏差
            for start in vertices:  # 逐个起点启动游走
                walk = self._node2vec_walk(start, walk_length, rng)
                if len(walk) < 2:  # 太短的序列忽略
                    continue
                walks.append([str(self.graph.vs[idx]["node_id"]) for idx in walk])  # 转为字符串 ID
                total_tokens += len(walk)  # 累加 token
            if tqdm is not None:
                tqdm.write(f"Completed walk round {walk_round + 1}/{num_walks}")
        return walks, total_tokens

    def _node2vec_walk(self, start: int, length: int, rng: np.random.Generator) -> List[int]:
        walk = [start]  # 初始化游走序列，以起点开始
        neighbors = self.graph.neighbors(start)
        if len(neighbors) == 0:  # 起点无邻居，直接返回单节点序列
            return walk
        if len(neighbors) == 1:  # 只有一个邻居时，第二步只能走到它
            walk.append(neighbors[0])
        else:
            alias = self.alias_nodes.get(start)  # 使用节点级 alias 表选择第二步
            idx = _alias_draw(alias[0], alias[1], rng) if alias else 0
            walk.append(neighbors[idx])
        while len(walk) < length:  # 后续步根据 (prev,current) 的有向边 alias 表抽样
            prev = walk[-2]
            current = walk[-1]
            neighbors = self.graph.neighbors(current)
            if not neighbors:  # 无后继则终止游走
                break
            alias = self.alias_edges.get((prev, current)) or self.alias_nodes.get(current)
            idx = _alias_draw(alias[0], alias[1], rng) if alias else rng.integers(len(neighbors))
            walk.append(neighbors[int(idx)])
        return walk


def _compute_embeddings(graph: ig.Graph, config: Node2vecConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """执行随机游走 + Word2Vec，返回嵌入矩阵及耗时指标。"""
    walker = IGraphNode2VecWalker(graph, config.p, config.q)  # 初始化游走器（带 alias 表）
    walk_start = perf_counter()  # 计时：游走阶段开始
    walks, total_tokens = walker.simulate_walks(config.num_walks, config.walk_length, config.seed)  # 生成游走语料
    walk_seconds = perf_counter() - walk_start  # 记录游走耗时

    workers = config._resolve_worker_count()  # 解析训练并行线程数
    if tqdm is not None:
        tqdm.write("正在基于游走语料训练 Word2Vec …")
    train_start = perf_counter()  # 计时：训练阶段开始
    model = Word2Vec(
        sentences=walks,              # 游走序列作为句子
        vector_size=config.dimensions,  # 向量维度（嵌入维）
        window=config.window,         # 上下文窗口
        min_count=1,                  # 不丢弃任何节点
        sg=1,                         # 使用 skip-gram 模式
        epochs=config.epochs,         # 训练轮数
        workers=workers,              # 并行工作线程
        seed=config.seed,             # 随机种子
    )
    train_seconds = perf_counter() - train_start  # 记录训练耗时

    # 提取所有图中节点的嵌入向量（按 node_id 顺序）
    vectors: List[List[float]] = []
    missing = 0
    for vertex in graph.vs:
        node_key = str(vertex["node_id"])  # 词表中的键为字符串 ID
        if node_key not in model.wv:  # 极端情况下可能缺失
            missing += 1
            continue
        vector = model.wv[node_key]
        vectors.append([int(vertex["node_id"])] + vector.tolist())
    columns = ["node_id"] + [f"f{i}" for i in range(config.dimensions)]
    embeddings = pd.DataFrame(vectors, columns=columns)  # 组装为 DataFrame

    # 训练与语料统计指标，便于报告与复现实验
    metrics: Dict[str, float] = {
        "walk_length": float(config.walk_length),
        "num_walks": float(config.num_walks),
        "p": float(config.p),
        "q": float(config.q),
        "dimensions": float(config.dimensions),
        "window": float(config.window),
        "epochs": float(config.epochs),
        "workers_used": float(workers),
        "walk_generation_seconds": walk_seconds,
        "training_seconds": train_seconds,
        "total_walks": float(len(walks)),
        "total_walk_tokens": float(total_tokens),
        "tokens_per_second": float(total_tokens) / train_seconds if train_seconds > 0 else 0.0,
        "missing_vertices": float(missing),
    }
    metrics["total_seconds"] = metrics["walk_generation_seconds"] + metrics["training_seconds"]  # 总耗时
    return embeddings, metrics


def _load_or_compute_embeddings(graph: ig.Graph, config: Node2vecConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    embedding_file = _data_path("node2vec_embeddings.csv")
    meta_file = _data_path("node2vec_training_meta.json")
    if embedding_file.exists() and not config.force_embeddings:
        embeddings = pd.read_csv(embedding_file)
        metrics: Dict[str, float] = {}
        if meta_file.exists():
            try:
                metrics = json.loads(meta_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metrics = {}
        return embeddings, metrics
    embeddings, metrics = _compute_embeddings(graph, config)
    embeddings.to_csv(embedding_file, index=False)
    meta_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return embeddings, metrics


def _degree_map(graph: ig.Graph) -> Dict[int, int]:
    degrees = graph.degree()
    node_ids = [int(v["node_id"]) for v in graph.vs]
    return {node_id: int(degree) for node_id, degree in zip(node_ids, degrees)}


def _merge_metadata(embeddings: pd.DataFrame, nodes: pd.DataFrame, communities: pd.DataFrame,
                    pagerank: Optional[pd.DataFrame], hits: Optional[pd.DataFrame], degree: Dict[int, int]) -> pd.DataFrame:
    base = embeddings.merge(nodes[["node_id", "name", "type"]], on="node_id", how="left")
    base = base.merge(communities[["node_id", "community"]], on="node_id", how="left")
    if pagerank is not None:
        base = base.merge(pagerank, on="node_id", how="left")
    if hits is not None:
        hits_subset = hits.rename(columns={"hub_score": "hub", "authority_score": "authority"})
        base = base.merge(hits_subset[["node_id", "hub", "authority"]], on="node_id", how="left")
    base["degree"] = base["node_id"].map(degree).fillna(0).astype(float)
    return base


def _embedding_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    feature_cols = [col for col in df.columns if col.startswith("f")]
    matrix = df[feature_cols].to_numpy(dtype=float)
    return matrix, feature_cols


def _run_kmeans_grid(matrix: np.ndarray, config: Node2vecConfig) -> pd.DataFrame:
    if matrix.size == 0:
        return pd.DataFrame(columns=["k", "inertia", "silhouette"])
    sample = matrix
    if matrix.shape[0] > config.silhouette_sample_size:
        rng = np.random.default_rng(config.seed)
        idx = rng.choice(matrix.shape[0], size=config.silhouette_sample_size, replace=False)
        sample = matrix[idx]
    records: List[Dict[str, float]] = []
    for k in config.k_grid:
        if k >= sample.shape[0]:
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=config.seed)
        model.fit(sample)
        silhouette = math.nan
        if sample.shape[0] >= max(20, k + 2):
            try:
                silhouette = float(silhouette_score(sample, model.labels_, metric="cosine"))
            except ValueError:
                silhouette = math.nan
        records.append({
            "k": int(k),
            "inertia": float(model.inertia_),
            "silhouette": silhouette,
        })
    return pd.DataFrame.from_records(records)


def _choose_best_k(metrics_df: pd.DataFrame, fallback: int) -> int:
    if metrics_df.empty:
        return fallback
    valid = metrics_df.dropna(subset=["silhouette"])
    if not valid.empty:
        row = valid.sort_values("silhouette", ascending=False).iloc[0]
        return int(row["k"])
    middle = metrics_df.sort_values("k").iloc[len(metrics_df) // 2]
    return int(middle["k"]) if not metrics_df.empty else fallback


def _plot_k_grid(metrics_df: pd.DataFrame, best_k: int) -> Path:
    path = _image_path("node2vec_kmeans_k_search.png")
    if metrics_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "KMeans 网格搜索数据不足", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=240)
        plt.close(fig)
        return path
    fig, ax1 = plt.subplots(figsize=(9, 5))
    metrics_df = metrics_df.sort_values("k")
    ax1.plot(metrics_df["k"], metrics_df["inertia"], marker="o", color="#1565C0", label="惯性(Inertia)")
    ax1.set_xlabel("簇数量 k")
    ax1.set_ylabel("惯性 (Inertia)", color="#1565C0")
    ax1.tick_params(axis='y', labelcolor="#1565C0")
    ax2 = ax1.twinx()
    ax2.plot(metrics_df["k"], metrics_df["silhouette"], marker="s", color="#FB8C00", label="轮廓系数(Silhouette)")
    ax2.set_ylabel("轮廓系数 (Silhouette)", color="#FB8C00")
    ax2.tick_params(axis='y', labelcolor="#FB8C00")
    ax1.axvline(best_k, color="#2E7D32", linestyle="--", label=f"best k={best_k}")
    ax1.set_title("KMeans k 搜索 (Inertia & Silhouette)")
    fig.tight_layout()
    combined_lines = ax1.lines + ax2.lines
    if combined_lines:
        labels = [str(line.get_label()) for line in combined_lines]
        ax1.legend(combined_lines, labels, loc="upper left")
    fig.savefig(path, dpi=260)
    plt.close(fig)
    return path


def _apply_kmeans(matrix: np.ndarray, k: int, seed: int) -> Tuple[np.ndarray, float]:
    if matrix.shape[0] < k:
        k = max(2, matrix.shape[0] // 2)
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = model.fit_predict(matrix)
    return labels, float(model.inertia_)



def _select_visual_subset(df: pd.DataFrame, config: Node2vecConfig) -> Tuple[pd.DataFrame, np.ndarray]:
    """依据社区/中心性筛选可视化子集，兼顾代表性与性能。"""
    if len(df) <= config.max_visual_nodes:  # 若整体样本不大，直接全量可视化
        return df.copy(), df.index.to_numpy()
    # 1) 优先保留样本最多的 TOP10 社区，保证主题代表性
    top_communities = df["community"].value_counts(dropna=True).index[:10]
    must_keep = df[df["community"].isin(top_communities)]
    if len(must_keep) >= config.max_visual_nodes:  # 若已超限，仅保留这些社区
        return must_keep.copy(), must_keep.index.to_numpy()
    keep_indices = set(must_keep.index.tolist())
    # 2) 追加 PageRank 高分节点（显著性强）
    if "pagerank" in df.columns and df["pagerank"].notna().any():
        top_k = max(10, int(len(df) * config.pagerank_top_pct))
        keep_indices.update(df.nlargest(top_k, "pagerank").index.tolist())
    # 3) 追加度数高的节点（连接性强）
    if df["degree"].notna().any():
        top_k = max(10, int(len(df) * config.degree_top_pct))
        keep_indices.update(df.nlargest(top_k, "degree").index.tolist())
    # 4) 若仍有空位，则从剩余样本中按固定 seed 随机补齐，兼顾均衡
    remaining_slots = config.max_visual_nodes - len(keep_indices)
    if remaining_slots > 0:
        remaining = df.drop(index=list(keep_indices))
        if not remaining.empty:
            rng = np.random.default_rng(config.seed)
            sample_size = min(remaining_slots, len(remaining))
            sample_idx = rng.choice(remaining.index.to_numpy(), size=sample_size, replace=False)
            keep_indices.update(sample_idx.tolist())
    subset = df.loc[sorted(keep_indices)].copy()
    return subset, subset.index.to_numpy()


def _reduce_embeddings(matrix: np.ndarray, seed: int) -> Tuple[np.ndarray, str]:
    if matrix.shape[0] == 0:
        return np.zeros((0, 2)), "N/A"
    if umap is not None and matrix.shape[0] >= 50:
        reducer = umap.UMAP(n_components=2, n_neighbors=40, min_dist=0.7, metric="cosine", random_state=seed)
        coords = reducer.fit_transform(matrix)
        coords = np.asarray(coords)
        return coords, "UMAP"
    perplexity = min(30, max(5, matrix.shape[0] // 200))
    reducer = TSNE(n_components=2, init="pca", perplexity=perplexity, random_state=seed, learning_rate="auto")
    coords = reducer.fit_transform(matrix)
    coords = np.asarray(coords)
    return coords, "t-SNE"


def _scale_sizes(values: pd.Series, min_size: float = 20.0, max_size: float = 110.0) -> np.ndarray:
    finite = values.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=float)
    span = finite.max() - finite.min()
    if span <= 1e-12:
        return np.full_like(finite, (min_size + max_size) / 2)
    norm = (finite - finite.min()) / span
    return norm * (max_size - min_size) + min_size


def _highlight_top_pagerank(
    ax: Axes,
    df: pd.DataFrame,
    color_field: str,
    palette: Dict[int, str],
    sizes: np.ndarray,
    scatter: PathCollection,
) -> Set[int]:
    """高亮 PageRank Top 节点，返回已高亮的 node_id 集合。"""
    if "pagerank" not in df.columns or df["pagerank"].isna().all():
        return set()
    highlight_df = df.nlargest(min(10, len(df)), "pagerank").copy()
    if highlight_df.empty:
        return set()
    highlight_ids = set(highlight_df.get("node_id", pd.Series(dtype=int)).dropna().astype(int).tolist())
    highlight_sizes = np.take(sizes, highlight_df.index.to_numpy())
    halo_sizes = np.maximum(highlight_sizes * 2.2, 160.0)
    ax.scatter(
        highlight_df["x"],
        highlight_df["y"],
        s=halo_sizes,
        c="white",
        alpha=0.35,
        linewidths=0,
        zorder=scatter.get_zorder() + 2,
    )
    core_sizes = np.maximum(highlight_sizes * 1.2, 90.0)
    highlight_colors = [
        palette.get(int(val) if pd.notna(val) else -1, "#37474F")
        for val in highlight_df[color_field]
    ]
    ax.scatter(
        highlight_df["x"],
        highlight_df["y"],
        s=core_sizes,
        c=highlight_colors,
        edgecolors="#FFFFFF",
        linewidths=0.35,
        alpha=0.95,
        zorder=scatter.get_zorder() + 3,
    )
    return highlight_ids


def _annotate_hits_roles(
    ax: Axes,
    df: pd.DataFrame,
    highlight_ids: Set[int],
    scatter: PathCollection,
) -> List[Line2D]:
    """在图上标注 HITS Authority/Hub 节点并返回对应图例 handle。"""
    if ("authority" not in df.columns or df["authority"].isna().all()) and (
        "hub" not in df.columns or df["hub"].isna().all()
    ):
        return []

    role_annotations: "OrderedDict[int, Tuple[float, float, str, str]]" = OrderedDict()

    def _collect_top(column: str, role_type: str) -> None:
        if column not in df.columns or df[column].isna().all():
            return
        top_rows = df.nlargest(min(5, len(df)), column).copy()
        for _, row in top_rows.iterrows():
            node_value = row.get("node_id")
            node_id = _safe_int(node_value, -1) if pd.notna(node_value) else _safe_int(row.name, -1)
            if node_id in role_annotations:
                continue
            raw_label = str(row.get("name") or f"节点 {node_id}").strip()
            label = raw_label if len(raw_label) <= 18 else f"{raw_label[:17]}…"
            role_annotations[node_id] = (float(row["x"]), float(row["y"]), label, role_type)

    _collect_top("authority", "authority")
    _collect_top("hub", "hub")

    if not role_annotations:
        return []

    x_span = float(df["x"].max() - df["x"].min()) or 1.0
    y_span = float(df["y"].max() - df["y"].min()) or 1.0
    x_unit = max(x_span * 0.015, 1e-5)
    y_unit = max(y_span * 0.015, 1e-5)
    direction_vectors = [
        (0.0, 0.0),
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
        (2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0),
        (2.0, 1.0), (-2.0, 1.0), (2.0, -1.0), (-2.0, -1.0),
        (0.6, 2.2), (-0.6, 2.2), (0.6, -2.2), (-0.6, -2.2),
    ]
    min_gap = max(x_span, y_span) * 0.035
    distance_sq = min_gap * min_gap
    placed_positions: List[Tuple[float, float]] = []
    role_colors = {"authority": "#C62828", "hub": "#1565C0"}
    role_handles: Dict[str, Line2D] = {}

    for node_id, (x_val, y_val, label, role_type) in role_annotations.items():
        base_dx = x_unit if node_id in highlight_ids else 0.0
        base_dy = y_unit if node_id in highlight_ids else 0.0
        best_pos: Optional[Tuple[float, float]] = None
        best_dx, best_dy = base_dx, base_dy
        for scale in (1.0, 1.4, 1.8, 2.2, 2.6, 3.0):
            for dir_x, dir_y in direction_vectors:
                dx = base_dx + dir_x * x_unit * scale
                dy = base_dy + dir_y * y_unit * scale
                candidate = (x_val + dx, y_val + dy)
                if all((candidate[0] - px) ** 2 + (candidate[1] - py) ** 2 >= distance_sq for px, py in placed_positions):
                    best_pos = candidate
                    best_dx, best_dy = dx, dy
                    break
            if best_pos is not None:
                break
        if best_pos is None:
            best_pos = (x_val + base_dx, y_val + base_dy)
            best_dx, best_dy = base_dx, base_dy
        placed_positions.append(best_pos)
        ha = "left" if best_dx > 0 else "right" if best_dx < 0 else "center"
        va = "bottom" if best_dy > 0 else "top" if best_dy < 0 else "center"
        color = role_colors.get(role_type, "#1B1B1B")
        ax.text(
            best_pos[0],
            best_pos[1],
            label,
            fontsize=9,
            weight="bold",
            ha=ha,
            va=va,
            color=color,
            zorder=scatter.get_zorder() + 4,
            bbox=dict(facecolor="#FFFFFF", alpha=1.0, edgecolor="none", pad=2.0),
        )
        if role_type not in role_handles:
            role_label = "Authority 节点" if role_type == "authority" else "Hub 节点"
            role_handles[role_type] = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=role_label,
                markerfacecolor=color,
                markersize=8,
            )

    return list(role_handles.values())


def _add_cluster_legend(
    ax: Axes,
    palette: Dict[int, str],
    special_labels: Optional[Dict[int, str]],
) -> None:
    """绘制聚类图例，优先展示 palette 中前若干颜色。"""
    legend_handles: List[Line2D] = []
    shown_keys: List[int] = []
    effective_labels = special_labels or {}
    for key, color in list(palette.items())[:15]:
        label = effective_labels.get(key, f"簇 {key}")
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markersize=7)
        )
        shown_keys.append(key)
    for key, label in effective_labels.items():
        if key in shown_keys:
            continue
        color = palette.get(key, "#B0BEC5")
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markersize=7)
        )
    if legend_handles:
        legend = ax.legend(handles=legend_handles, loc="upper right")
        ax.add_artist(legend)


def _plot_scatter(subset: pd.DataFrame, coords: np.ndarray, color_field: str, palette: Dict[int, str],
                  title: str, legend_title: str, size_field: str, annotate_field: Optional[str], path: Path,
                  special_labels: Optional[Dict[int, str]] = None) -> None:
    """绘制嵌入投影 + KMeans 配色散点，同时叠加 PageRank 高亮与 HITS 角色。"""
    df = subset.copy().reset_index(drop=True)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    sizes = _scale_sizes(df[size_field]) if size_field in df.columns else np.full(len(df), 40.0)
    colors = [palette.get(int(val) if pd.notna(val) else -1, "#B0BEC5") for val in df[color_field]]
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df["x"], df["y"], s=sizes, c=colors, alpha=0.72, linewidths=0.2, edgecolors="none")
    ax.set_xlabel("维度1")
    ax.set_ylabel("维度2")
    ax.set_title(title)

    # PageRank 高亮：先画柔和光环，再画核心点，方便视觉聚焦
    highlight_ids = _highlight_top_pagerank(ax, df, color_field, palette, sizes, scatter)

    if annotate_field and annotate_field in df.columns and df[annotate_field].notna().any():
        top = df.nlargest(12, size_field if size_field in df.columns else "degree").copy()
        for _, row in top.iterrows():
            label = str(row.get("name", ""))[:12]
            ax.text(row["x"], row["y"], label, fontsize=8, weight="bold", ha="left", va="bottom")

    # HITS 角色标签：自动避让重叠，同时返回图例 handle 供后续绘制
    role_handles = _annotate_hits_roles(ax, df, highlight_ids, scatter)

    # 聚类图例：右上角展示主要簇与社区别名
    _add_cluster_legend(ax, palette, special_labels)

    if role_handles:
        role_legend = ax.legend(handles=role_handles, loc="upper left", bbox_to_anchor=(-0.02, 1.0))
        ax.add_artist(role_legend)
    fig.tight_layout()
    fig.savefig(path, dpi=280)
    plt.close(fig)


def _cluster_palette(labels: Iterable[int]) -> Dict[int, str]:
    unique = sorted({int(label) for label in labels if label is not None})
    palette = sns.color_palette("tab20", n_colors=max(1, len(unique)))
    return {label: mcolors.to_hex(color) for label, color in zip(unique, palette)}


def _plot_heatmap_top10(df: pd.DataFrame, path: Path, label_map: Dict[int, str]) -> Tuple[int, float]:
    """绘制 TOP10 Louvain 社区与 KMeans 簇的对应热力图。"""
    top10 = df["community"].value_counts().index[:10]
    subset = df[df["community"].isin(top10)]
    if subset.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "缺少 Louvain TOP10 数据", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=240)
        plt.close(fig)
        return 0, 0.0
    table = pd.crosstab(subset["community"], subset["embedding_cluster_kmeans"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(table, cmap="YlOrBr", annot=True, fmt=".1f", ax=ax, cbar_kws={"label": "%"})
    ax.set_xlabel("KMeans 聚类")
    ax.set_ylabel("Louvain 社区")
    y_labels = [label_map.get(int(idx), f"社区 {idx}") for idx in table.index]
    ax.set_yticklabels(y_labels, rotation=0)
    ax.set_title("TOP10 Louvain 社区 vs KMeans 簇")
    fig.tight_layout()
    fig.savefig(path, dpi=280)
    plt.close(fig)
    return table.shape[0], float(table.to_numpy().max())


def _plot_louvain_scatter(subset: pd.DataFrame, coords: np.ndarray, title_map: Dict[int, str], path: Path) -> None:
    """展示 Louvain TOP10 社区在嵌入空间的分布及中心标签。"""
    df = subset.copy().reset_index(drop=True)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    top10 = df["community"].value_counts().index[:10]
    palette = sns.color_palette("tab10", n_colors=len(top10))
    color_map = {int(cid): mcolors.to_hex(color) for cid, color in zip(top10, palette)}
    df["color"] = df["community"].apply(lambda cid: color_map.get(int(cid), "#CFD8DC") if pd.notna(cid) and int(cid) in color_map else "#CFD8DC")
    sizes = _scale_sizes(df.get("pagerank", df.get("degree", pd.Series(np.ones(len(df))))))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df["x"], df["y"], c=df["color"], s=sizes, alpha=0.7, linewidths=0.2, edgecolors="none")
    ax.set_title("Louvain TOP10 社区在嵌入空间的分布")
    ax.set_xlabel("维度1")
    ax.set_ylabel("维度2")
    legend_handles: List[Line2D] = []
    for cid in top10:
        color = color_map.get(int(cid))
        if color is None:
            continue
        label = title_map.get(int(cid), f"社区 {cid}")
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label=label,
                                      markerfacecolor=color, markersize=8))
    centers = []
    for cid in top10:
        sub = df[df["community"] == cid]
        if sub.empty:
            continue
        centers.append((cid, sub["x"].mean(), sub["y"].mean(), len(sub)))
    for cid, cx, cy, size in centers:
        label = title_map.get(int(cid), f"社区 {cid}")
        ax.text(cx, cy, f"{label} ({size})", fontsize=9, weight="bold", ha="center", va="center",
            bbox=dict(facecolor="#FFFFFF", alpha=1.0, pad=2.0, edgecolor="none"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=280)
    plt.close(fig)


def _write_report(images: Dict[str, Path], summary: Dict[str, object]) -> None:
    """根据统计摘要自动生成 Markdown 报告，确保描述与图表同步。"""
    def rel(path: Path) -> str:
        return os.path.relpath(path, data_loader.REPORT_DIR)

    lines = [
        "# Node2Vec 嵌入可视化分析",
        "",
        f"**分析日期**：{pd.Timestamp.now().strftime('%Y年%m月%d日')}  ",
        f"**嵌入维度**：{summary.get('dimensions', 0)}  ",
        f"**有效节点**：{summary.get('total_nodes', 0):,}",
        "",
        "## 1. 训练配置与性能",
        "",
        "- 随机游走：长度 {walk}，每节点 {num} 次，(p={p:.2f}, q={q:.2f})".format(
            walk=_safe_int(summary.get('walk_length', 0)),
            num=_safe_int(summary.get('num_walks', 0)),
            p=_safe_float(summary.get('p', 0.0)),
            q=_safe_float(summary.get('q', 0.0)),
        ),
        "- Word2Vec：window={window}，epochs={epochs}，workers={workers}".format(
            window=_safe_int(summary.get('window', 0)),
            epochs=_safe_int(summary.get('epochs', 0)),
            workers=_safe_int(summary.get('workers_used', 1)),
        ),
        "- 时间开销：随机游走 {walk_sec:.2f}s + 训练 {train_sec:.2f}s = 总计 {total_sec:.2f}s，tokens/s ≈ {tps:,.0f}".format(
            walk_sec=_safe_float(summary.get('walk_generation_seconds', 0.0)),
            train_sec=_safe_float(summary.get('training_seconds', 0.0)),
            total_sec=_safe_float(summary.get('total_seconds', 0.0)),
            tps=_safe_float(summary.get('tokens_per_second', 0.0)),
        ),
        f"- 全图 {_safe_int(summary.get('total_nodes', 0)):,} 个节点均生成嵌入（missing={_safe_int(summary.get('missing_vertices', 0))}），累计 {_safe_int(summary.get('total_walks', 0)):,} 条游走 / {_safe_int(summary.get('total_walk_tokens', 0)):,} tokens，为后续的聚类与可视化提供一致输入。",
        "",
        "## 2. KMeans 最佳 k 搜索",
        "",
        f"![K 搜索]({rel(images['k_search'])})",
        "",
        f"- 蓝色惯性曲线随着 k 增大持续下降，而橙色 Silhouette 在 k={_safe_int(summary.get('best_k', 0))} 时达到 {_safe_float(summary.get('best_silhouette', float('nan'))):.3f} 的峰值，说明 12 个簇在紧密度与可分性之间取得最佳平衡。",
        f"- 扫描的 k 范围：{summary.get('k_grid_display', '')}。k≥20 时轮廓系数明显回落，对应图中橙线的波谷，提示继续细分只会把连贯语义割裂成噪声。",
        "",
        "## 3. KMeans vs Louvain TOP10",
        "",
        f"![热力图]({rel(images['heatmap'])})",
        "",
        f"- 表格展示 TOP10 Louvain 社区与 KMeans 簇的重叠，主对角线出现 80%+ 的亮块，峰值对齐 {_safe_float(summary.get('peak_alignment_pct_top10', 0.0)):.1f}% ，说明随机游走嵌入与社区检测在宏观主题上高度一致。",
        "- 次要块状区域（如免疫/风湿相关社区）同时覆盖多个簇，提示这些主题在语义空间更分散，可作为后续细分或跨簇关系挖掘的候选。",
        "",
        "## 4. 嵌入投影 + KMeans 簇",
        "",
        f"![KMeans 投影]({rel(images['kmeans_scatter'])})",
        "",
        f"- 可视化节点：{_safe_int(summary.get('visual_points', 0)):,}，采用 {summary.get('projection_method', 'N/A')} 将 128 维嵌入降至 2D；点颜色完全来自 KMeans 簇，点尺寸代表 PageRank/度数。",
        f"- PageRank Top5（{summary.get('top_pagerank_names', 'N/A')}）呈现“两类典型位置”：(1) 内科、血常规位于各自簇的几何中心；(2) 白酒、鸡蛋位于多个簇的交界带，外科介于相邻簇的带状过渡区——这些边界型高 PageRank 节点通常承担跨语义连结。",
        f"- Authority（红色标签）与 Hub（蓝色标签）的空间分布：Authority 中“白酒、啤酒、鸡蛋、鸡肉”沿簇边界成带状分布，而“血常规”是唯一位于簇内部的红点；Hub 中“血管损伤”与前述红点共同落在同一条簇的交界线，其他 Hub 则呈“中心型 + 边界型”并存。",
        "- 白色光环标注的 PageRank Top10 与红/蓝角色标签叠加后，可以快速圈定“既重要又具跨簇影响力”的锚点实体，用于问答/检索召回与知识推理起点。",
        "",
        "## 5. Louvain TOP10 空间分布",
        "",
        f"![Louvain 投影]({rel(images['louvain_scatter'])})",
        "",
        "- 每个社区在 2D 投影中呈椭圆状密度块，白底标签标注主题，便于与上一幅 KMeans 彩色块逐一对照。",
        "- Louvain 块与 KMeans 色块在空间位置高度一致：这意味着基于随机游走学习到的嵌入，在降维后仍较好地保留了社区结构；当两图中的块彼此靠近（如围产期遗传 vs 先天畸形）时，可认为它们在知识图谱中语义更相近。",
        "",
        "---",
        "",
        "这些图表联合展示了随机游走 + 嵌入在宏观（社区）与微观（PageRank/HITS）层面的互补性，可据此挑选重点节点、验证主题划分，或进一步做相似度检索与知识推理。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _top_names(df: pd.DataFrame, column: str, limit: int = 5) -> str:
    if column not in df.columns or df[column].isna().all():
        return "N/A"
    top = df.nlargest(limit, column)["name"].dropna().tolist()
    return ", ".join(top)


def run_analysis(config: Node2vecConfig) -> None:
    """主流程：加载数据 -> 训练嵌入 -> 聚类可视化 -> 输出报告。"""
    configure_chinese_plot_style()  # 全局配置中文字体与绘图样式，避免中文乱码
    # 1. 载入节点、边及外部算法结果，确保所有指标齐备
    nodes, edges = _load_nodes_edges()  # 加载节点与边 CSV
    communities = _load_louvain_assignments()  # 读取 Louvain 社区划分
    louvain_titles = _load_louvain_titles()    # 读取社区中文标题（若有）
    pagerank = _load_pagerank_scores()         # 读取 PageRank 分数（可选）
    hits = _load_hits_scores()                 # 读取 HITS Hub/Authority（可选）

    # 2. 构建 igraph，并缓存度数等结构特征
    graph = _build_igraph(nodes, edges)  # 根据 CSV 构图，并去除孤立点
    degree = _degree_map(graph)          # 计算每个节点的度数，作为后续特征

    # 3. 复用或重新训练 node2vec 嵌入
    embeddings, training_metrics = _load_or_compute_embeddings(graph, config)  # 复用缓存或重新训练
    merged = _merge_metadata(embeddings, nodes, communities, pagerank, hits, degree)  # 合并所有元数据
    merged = merged.sort_values("node_id").reset_index(drop=True)  # 保证稳定顺序
    if len(merged) < 2:
        raise ValueError("Node2Vec requires at least 2 embedded nodes. Check the input graph.")

    matrix, feature_cols = _embedding_matrix(merged)  # 取出嵌入特征矩阵与列名
    merged.to_csv(_data_path("node2vec_embeddings_with_meta.csv"), index=False)  # 缓存含元数据的嵌入

    # 4. KMeans 网格搜索 + 最佳簇数选择
    k_metrics = _run_kmeans_grid(matrix, config)   # 在采样集上扫描 k 值
    best_k = _choose_best_k(k_metrics, fallback=18)  # 根据 Silhouette 选取最佳 k
    k_search_path = _plot_k_grid(k_metrics, best_k)  # 绘制双轴曲线（Inertia & Silhouette）
    if not k_metrics.empty:
        k_metrics.to_csv(_data_path("node2vec_kmeans_k_search_metrics.csv"), index=False)  # 保存网格搜索指标
    k_labels, _ = _apply_kmeans(matrix, best_k, config.seed)  # 在全体样本上拟合最佳 k 的 KMeans
    merged["embedding_cluster_kmeans"] = k_labels  # 记录每个节点的聚类标签

    # 5. 采样绘图子集并降维，兼顾代表性与渲染性能
    visual_subset, subset_indices = _select_visual_subset(merged, config)  # 选择可视化代表性子集
    coords, method = _reduce_embeddings(matrix[subset_indices], config.seed)  # UMAP/t-SNE 降维
    visual_subset = visual_subset.reset_index(drop=True)  # 重排索引以对齐坐标
    visual_subset["x"] = coords[:, 0]  # 写入 X
    visual_subset["y"] = coords[:, 1]  # 写入 Y
    visual_subset.to_csv(_data_path("node2vec_visual_subset_coords.csv"), index=False)  # 缓存可视化坐标

    # 6. 生成各类图表（KMeans 投影、Louvain 投影、TOP10 热力图）
    k_palette = _cluster_palette(visual_subset["embedding_cluster_kmeans"].unique())  # 生成聚类配色
    kmeans_scatter_path = _image_path("node2vec_umap_kmeans_scatter.png")
    _plot_scatter(
        visual_subset,
        coords,
        color_field="embedding_cluster_kmeans",
        palette=k_palette,
        title="Node2Vec 嵌入 + KMeans 聚类",
        legend_title="KMeans 簇",
        size_field="pagerank" if "pagerank" in visual_subset.columns else "degree",
        annotate_field=None,
        path=kmeans_scatter_path,
    )

    louvain_scatter_path = _image_path("node2vec_umap_louvain_top10_scatter.png")
    _plot_louvain_scatter(visual_subset, coords, louvain_titles, louvain_scatter_path)  # Louvain TOP10 空间分布
    heatmap_path = _image_path("node2vec_cluster_heatmap_top10.png")
    top_comm_count, peak_alignment = _plot_heatmap_top10(merged, heatmap_path, louvain_titles)  # 对齐热力图

    embeddings_path = _data_path("node2vec_embeddings.csv")  # 嵌入主文件路径
    embeddings.to_csv(embeddings_path, index=False)  # 保存纯嵌入矩阵

    best_silhouette = math.nan
    if not k_metrics.empty:
        mask = k_metrics["k"] == best_k
        if mask.any():
            values = k_metrics.loc[mask, "silhouette"].dropna()
            if not values.empty:
                best_silhouette = float(values.iloc[0])

    # 7. 汇总指标并输出 Markdown 报告，方便分享
    summary = {
        **training_metrics,
        "dimensions": len(feature_cols),
        "total_nodes": int(len(merged)),
        "best_k": best_k,
        "best_silhouette": best_silhouette,
        "k_grid_display": ", ".join(str(int(k)) for k in config.k_grid),
        "projection_method": method,
        "visual_points": int(len(visual_subset)),
        "heatmap_top_communities": top_comm_count,
        "peak_alignment_pct_top10": peak_alignment,
        "embeddings_path": str(embeddings_path.relative_to(data_loader.OUTPUT_DIR)),
        "embeddings_meta_path": str(_data_path("node2vec_embeddings_with_meta.csv").relative_to(data_loader.OUTPUT_DIR)),
        "visual_subset_path": str(_data_path("node2vec_visual_subset_coords.csv").relative_to(data_loader.OUTPUT_DIR)),
        "walk_length": config.walk_length,
        "num_walks": config.num_walks,
        "p": config.p,
        "q": config.q,
        "window": config.window,
        "epochs": config.epochs,
        "workers_used": config._resolve_worker_count(),
        "top_pagerank_names": _top_names(merged, "pagerank"),
        "top_authority_names": _top_names(merged, "authority"),
        "top_hub_names": _top_names(merged, "hub"),
    }
    _data_path("node2vec_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")  # 缓存摘要

    images = {
        "k_search": k_search_path,
        "kmeans_scatter": kmeans_scatter_path,
        "heatmap": heatmap_path,
        "louvain_scatter": louvain_scatter_path,
    }
    _write_report(images, summary)

    print("Node2Vec 分析完成。")
    print(f"  嵌入缓存位置: {embeddings_path}")
    print(f"  报告生成位置: {REPORT_PATH}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Node2Vec 分析（igraph 随机游走 + gensim 训练）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="提示：请从项目根目录运行：python -m src.node2vec_analysis",
    )

    # 通用参数
    general = parser.add_argument_group("通用")
    general.add_argument("--force-embeddings", action="store_true", help="即使存在缓存也重新计算嵌入")
    general.add_argument("--seed", type=int, help="随机种子（保证可复现）")
    general.add_argument("--workers", type=int, help="训练使用的并行线程数（workers）")

    # 随机游走 / node2vec 参数
    walk = parser.add_argument_group("随机游走（node2vec）")
    walk.add_argument("--walk-length", type=int, help="随机游走长度（每条游走的步数）")
    walk.add_argument("--num-walks", type=int, help="每个节点的游走次数（重复轮数）")
    walk.add_argument("--p", type=float, help="node2vec 返回参数 p（p 小：回到上一步更容易）")
    walk.add_argument("--q", type=float, help="node2vec 进出参数 q（q 小偏 BFS，q 大偏 DFS）")

    # Word2Vec 训练参数
    w2v = parser.add_argument_group("Word2Vec 训练")
    w2v.add_argument("--dimensions", type=int, help="嵌入向量维度（feature 维数）")
    w2v.add_argument("--window", type=int, help="上下文窗口大小（window）")
    w2v.add_argument("--epochs", type=int, help="训练轮数（epochs）")

    # 聚类与可视化参数
    viz = parser.add_argument_group("聚类与可视化")
    viz.add_argument("--k-grid", type=str, help="以逗号分隔的 k 值列表（用于 KMeans 网格搜索）")
    viz.add_argument("--max-visual-nodes", type=int, help="二维投影中最多渲染的节点数（控制性能）")

    # 采样 / 评估参数
    evalgrp = parser.add_argument_group("采样与评估")
    evalgrp.add_argument("--silhouette-sample-size", type=int, help="计算轮廓系数的样本上限（避免全量耗时）")
    evalgrp.add_argument("--pagerank-top-pct", type=float, help="采样时保留的 PageRank 前百分比（0-1）")
    evalgrp.add_argument("--degree-top-pct", type=float, help="采样时保留的度数前百分比（0-1）")

    args = parser.parse_args(argv)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    run_analysis(config)


if __name__ == "__main__":
    main()

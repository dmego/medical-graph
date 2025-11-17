# 医疗知识图谱分析平台 - AI 编程助手指南

## 快速上手（AI 代理必读）
- 目标：基于医疗知识图谱（约 4.4 万节点、30 万边）运行独立算法模块，自动产出图表与 Markdown 报告。
- 核心依赖：`networkx`、`pandas`、`numpy`、`matplotlib`、`seaborn`、`python-louvain`、`scikit-learn`、`node2vec`（可选 `umap-learn`）。

- 统一数据/路径入口：始终通过 `src/data_loader.py` 访问数据与路径常量：
    - `load_nodes()` / `load_edges()` 加载 `data/*.csv`
    - `resolve_output(relpath)` 创建并返回 `output/` 下的完整输出路径
    - 重要常量：`PROJECT_DIR`、`DATA_DIR`、`OUTPUT_DIR`、`REPORT_DIR`

- 绘图前必须调用中文样式：
    - `from src.set_chinese_plot import configure_chinese_plot_style`
    - `configure_chinese_plot_style()` 修复中文与负号显示，统一主题

- 标准图构建范式（有向图）：
    - 使用 `nx.DiGraph()`；节点属性含 `name`、`type`；边属性含 `rel_type`
    - 视情况移除孤立点：`G.remove_nodes_from(list(nx.isolates(G)))`

- 运行方式（必须从项目根目录，以模块运行）：
    - `python -m src.graph_analysis`
    - `python -m src.pagerank_analysis`
    - `python -m src.hits_analysis`
    - `python -m src.louvain_analysis`
    - `python -m src.node2vec_analysis [--force-embeddings]`

- 输出规范（自动覆盖旧结果）：
    - 中间结果：`output/data/<algo>/...`（CSV/JSON）
    - 图表：`output/images/<algo>/...`（PNG, 建议 `dpi=240/300`）
    - 报告：`reports/<algo>_analysis.md`
    - 报告中引用图片使用相对路径（相对 `reports/`）：`os.path.relpath(image, REPORT_DIR)`

- Node2Vec 要点：
    - 缓存：`output/data/node2vec/node2vec_embeddings.csv` 存在即复用；`--force-embeddings` 重新训练
    - 降维优先 UMAP；缺失时自动回退 t‑SNE；KMeans 聚类并产出 Louvain 对齐分析

- Louvain 要点：
    - 先转为加权无向图（权重=多重边计数），再 `community_louvain.best_partition`
    - 生成社区级可视化与标题（`jieba` 高频词，`wordcloud` 如可用）

- 项目特定约定与易错点：
    - 路径一律用 `resolve_output()`，不要手动 `os.path.join('output', ...)`
    - 运行目录必须是仓库根目录；不要在 `src/` 下直接运行脚本
    - `data/` 为只读；所有写入到 `output/` 与 `reports/`
    - 大图运算较慢，调试可先用子图（如仅 `acompany_with` 关系）
    - 统计学展示：小数用科学计数法，必要时 `FuncFormatter(lambda v, _: f"{v:.2e}")`
    - 组件分析偏好强连通分量（SCC），最短路采用采样以控成本
    - Python 3.9+ 类型注解风格，保持与现有风格一致

## 项目概览

这是一个**医疗知识图谱分析平台**，使用 Python 分析包含 4.4 万实体和 30 万关系的医疗知识图谱。核心功能包括基础图分析、PageRank、HITS 和 Louvain 社区发现算法，自动生成可视化图表和 Markdown 分析报告。

**关键特性**：
- 7 类节点：Disease（疾病）、Drug（药品）、Food（食物）、Check（检查）、Department（科室）、Producer（药品厂商）、Symptom（症状）
- 10 种关系类型：如 `acompany_with`（并发）、`common_drug`（通用药）、`has_symptom`（有症状）等
- 所有分析模块都是**独立可运行的**，从项目根目录以 `-m` 方式执行

## 架构与数据流

### 1. 数据加载层 (`src/data_loader.py`)
**这是整个项目的基础设施**，所有模块通过它访问数据和路径：

```python
from src.data_loader import load_nodes, load_edges, OUTPUT_DIR, REPORT_DIR
nodes_df = load_nodes()  # 加载 data/nodes.csv
edges_df = load_edges()  # 加载 data/edges.csv
```

**关键路径常量**：
- `PROJECT_DIR`: 项目根目录
- `DATA_DIR`: 原始数据目录 (`data/`)
- `OUTPUT_DIR`: 所有生成文件根目录 (`output/`)
- `REPORT_DIR`: Markdown 报告目录 (`reports/`)

**重要函数**：
- `resolve_output(path)`: 自动在 `output/` 下创建目录并返回完整路径
- `load_nodes(columns=None)`: 可选择性加载节点属性列
- `load_edges(columns=None)`: 可选择性加载边属性列

### 2. 中文图表配置 (`src/set_chinese_plot.py`)
**所有分析模块在绘图前必须调用**：

```python
from src.set_chinese_plot import configure_chinese_plot_style
configure_chinese_plot_style()
```

- 配置跨平台中文字体回退链（macOS/Windows/Linux）
- 修复 Unicode 负号显示问题
- 统一 seaborn 样式和字体大小

### 3. 分析模块结构模式
所有分析脚本 (`graph_analysis.py`, `pagerank_analysis.py`, `hits_analysis.py`, `louvain_analysis.py`, `node2vec_analysis.py`) 遵循相同模式：

1. **构建图** → NetworkX 有向图 (`nx.DiGraph`)
2. **执行算法** → 计算得分/社区划分
3. **保存数据** → CSV/JSON 到 `output/data/<algorithm>/`
4. **生成可视化** → PNG 到 `output/images/<algorithm>/`
5. **生成报告** → Markdown 到 `reports/<algorithm>_analysis.md`

## 关键编码约定

### 图构建规范
```python
# 标准图构建模式（有向图）
G = nx.DiGraph()
for _, row in nodes_df.iterrows():
    G.add_node(row['node_id'], name=row['name'], type=row['type'])
for _, row in edges_df.iterrows():
    G.add_edge(row['src_id'], row['dst_id'], rel_type=row['rel_type'])

# 移除孤立节点是常见操作
G.remove_nodes_from(list(nx.isolates(G)))
```

### 路径处理规范
**永远使用 `data_loader.resolve_output()` 而不是手动拼接路径**：

```python
# ✅ 正确
output_path = data_loader.resolve_output("images/pagerank/distribution.png")

# ❌ 错误 - 不要手动拼接路径
output_path = os.path.join("output", "images", "pagerank", "distribution.png")
```

### 类型注解风格
项目使用现代 Python 类型注解（支持 Python 3.9+）：

```python
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, cast

def analyze_degree(G: nx.DiGraph) -> Dict[str, float]:
    degrees: List[int] = [d for n, d in G.degree()]  # type: ignore
    return {"mean": np.mean(degrees)}
```

### 报告生成模式
**Markdown 报告使用相对路径引用图片**（相对于 `reports/` 目录）：

```python
def _write_markdown(output_path: Path, image_path: Path):
    rel_path = os.path.relpath(image_path, data_loader.REPORT_DIR)
    lines = [
        "# 分析报告",
        f"![图表]({rel_path})",  # 例如: ../output/images/pagerank/distribution.png
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
```

## 开发工作流

### 运行分析脚本
**必须从项目根目录以模块方式运行**：

```bash
# ✅ 正确 - 从根目录运行
python -m src.graph_analysis
python -m src.pagerank_analysis
python -m src.hits_analysis
python -m src.louvain_analysis
python -m src.node2vec_analysis  # 生成 embeddings 及四张可视化

# ❌ 错误 - 不要直接运行脚本
cd src && python graph_analysis.py
```

### 环境设置
```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 2. 安装依赖
pip install -r requirements.txt
# 依赖包含 scikit-learn、node2vec；如需更平滑的降维可手动安装 umap-learn (可选，需 Python <3.14)
```

### 调试建议
- **大图分析很慢**：在开发时可以先用子图测试（如只用 `acompany_with` 关系）
- **中文显示问题**：检查是否调用了 `configure_chinese_plot_style()`
- **路径错误**：确保从项目根目录运行，而不是 `src/` 目录

## 算法特定知识

### PageRank 分析
- 使用 `nx.pagerank(G, alpha=0.85, max_iter=100)` 计算
- **疾病类型最重要**：Top 200 节点中疾病和症状占 50%+
- 包含 **科室分布雷达图**（双指标：疾病数量 + 平均 PageRank）
- 自动识别"异常值"节点（> 3×均值的高分节点）

### HITS 分析
- 区分 **Hub（枢纽）** 和 **Authority（权威）**
- Hub: 连接多个重要节点的"信息聚合者"（如综合检查项目）
- Authority: 被多个重要节点指向的"知识核心"（如核心疾病）
- 包含与 PageRank 的差异分析（z-score 标准化比较）

### Louvain 社区发现
- 使用 `python-louvain` 库（`community.community_louvain.best_partition`）
- 先转换为**加权无向图**（边权重 = 多重边数量）
- **自动生成社区标题**：使用 jieba 分词提取高频词
- 包含社区级图（community-level graph）可视化
- 使用 WordCloud 生成社区词云（如果库可用）

### 基础图分析
- **度分布**：双对数坐标系，检测幂律特征（无标度网络）
- **聚类系数**：按度计算平均聚类系数 C(k)
- **路径长度**：采样计算（默认 10,000 对节点），避免全局计算开销
- **连通分量**：使用**强连通分量**（SCC）而非弱连通分量

### Node2Vec 嵌入综合分析
- `node2vec_analysis.py` 会缓存 `output/data/node2vec/node2vec_embeddings.csv`，存在时默认复用（可传 `force_embeddings=True` 重新计算）
- 依赖 `scikit-learn`、`node2vec`；若安装了 `umap-learn`（可选）则优先用 UMAP，否则自动降级使用 t-SNE
- 生成四类可视化：整体嵌入分布、语义相似网络、嵌入聚类 vs Louvain 热力图、典型社区局部嵌入
- 相似网络通过余弦近邻筛选 top 1% 相似度边，并标注跨社区桥接节点
- 聚类默认使用 KMeans（n≈18），报告中记录最大对齐占比与覆盖社区数量

## 常见任务示例

### 添加新的图算法分析
1. 在 `src/` 下创建 `your_algorithm_analysis.py`
2. 导入基础设施：
   ```python
   from src.data_loader import load_nodes, load_edges, OUTPUT_DIR, REPORT_DIR, resolve_output
   from src.set_chinese_plot import configure_chinese_plot_style
   ```
3. 遵循标准流程：加载数据 → 构建图 → 计算 → 可视化 → 生成报告
4. 输出路径使用 `resolve_output("data/your_algo/scores.csv")`
5. 在 `if __name__ == "__main__":` 下添加主函数

### 修改可视化样式
**统一在 `set_chinese_plot.py` 中修改**，避免分散在各个模块：

```python
# 修改全局字体大小
plt.rcParams["font.size"] = 12  # 当前是 10

# 修改图表主题
sns.set_theme(style="darkgrid")  # 当前是 whitegrid
```

### 处理新的节点/关系类型
在 `data_loader.py` 中定义常量：

```python
RELATION_TYPES = (
    "acompany_with", "belongs_to", "common_drug", 
    "do_eat", "drugs_of", "has_symptom",
    # 添加新关系类型
)
```

## 数据文件格式

### `data/nodes.csv`
```
node_id,name,type,other_attrs
1,高血压,Disease,"{\"cause\":\"...\",\"cure_department\":[\"心内科\"],...}"
2,阿司匹林,Drug,
```

### `data/edges.csv`
```
src_id,dst_id,rel_type,other_attrs
1,2,common_drug,"{\"name\":\"常用药\"}"
1,3,has_symptom,"{\"name\":\"症状\"}"
```

## 注意事项

- **不要修改原始数据**：`data/` 目录是只读的
- **输出文件自动覆盖**：重新运行分析会覆盖 `output/` 和 `reports/` 中的文件
- **图片 DPI 标准**：使用 `dpi=240` 或 `dpi=300` 确保清晰度
- **科学计数法**：小数值（如 PageRank）使用 `FuncFormatter(lambda v, _: f"{v:.2e}")`
- **JSON 属性解析**：`other_attrs` 字段是 JSON 字符串，需要 `json.loads()` 解析

## 项目文件组织

```
medical-graph/
├── data/               # 原始数据（只读）
│   ├── nodes.csv
│   └── edges.csv
├── src/                # 源代码（所有分析模块）
│   ├── data_loader.py      # 核心：数据加载和路径管理
│   ├── set_chinese_plot.py # 核心：中文图表配置
│   ├── graph_analysis.py   # 基础图分析
│   ├── pagerank_analysis.py
│   ├── hits_analysis.py
│   ├── louvain_analysis.py
│   └── node2vec_analysis.py
├── output/             # 生成的数据和图表
│   ├── data/          # CSV/JSON 中间结果
│   └── images/        # PNG 可视化图表
├── reports/            # Markdown 分析报告
├── lib/                # 前端库（未使用）
├── requirements.txt
└── README.md
```

---

**关键原则**：保持模块独立性、统一路径管理、遵循中文可视化规范、从根目录运行。

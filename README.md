# 医疗知识图谱分析工具

这是一个用于分析医疗知识图谱基本属性的Python工具，可以计算和可视化图的度分布、路径长度、聚类系数和连通分量等关键指标。

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)]()

## 📊 项目概览

本项目对包含 **44,112个节点** 和 **291,165条边** 的医疗知识图谱进行了全面分析，生成了详细的网络拓扑报告和可视化图表。

### 核心发现

- 🎯 **无标度网络**: 幂律指数 γ=1.53，符合复杂网络特征
- 🌐 **小世界网络**: 平均路径长度仅4.42步，信息传递高效
- ⭐ **超级节点**: 最大度达2,755，存在医疗领域的核心实体
- 🔗 **完全连通**: 100%节点在同一弱连通分量中
- 📈 **低聚类**: 聚类系数0.0062，呈"以疾病为中心"的星型结构

### 完整分析报告

📄 **[查看完整分析报告](reports/medical_graph_complete_analysis.md)**

报告包含详细的数据解读和业务含义说明，类似MSN网络分析：
- 度分布：平均12.54，说明平均每个医疗实体关联约13个其他实体
- 聚类系数：0.0062，体现星型结构，以疾病为中心辐射
- 路径长度：平均4.42步，任意两实体最多8步可达
- 连通分量：强连通覆盖0.27%，体现医疗关系的方向性

## 🎨 功能特性

✅ **完整图分析**
- 度分布分析（双对数坐标 + 幂律拟合）
- 聚类系数分析（按度汇总 + 幂律拟合）
- 路径长度分布（采样法 + 小世界网络检测）
- 连通分量分析（强连通分量 + 弱连通分量）

✅ **子图分析**
- 支持按关系类型分析子图（当前配置：仅分析 `acompany_with` 并发疾病关系）
- 每个子图进行独立的四大属性分析
- 自动跳过节点数过少的子图

✅ **专业可视化**
- 所有图表采用对数坐标系展示幂律分布
- 中英文双语标注，便于理解
- 酒红色空心圆标记点，美观清晰
- 全局 ASCII 负号格式化，避免字体警告
- 高分辨率输出（300 DPI）

## 📁 项目结构

```
medical-codex/
├── data/                          # 数据文件
│   ├── nodes.csv                  # 节点数据（44,112个节点）
│   └── edges.csv                  # 边数据（291,165条边）
├── src/                           # 源代码
│   ├── graph_analysis.py          # 主分析程序
│   └── fonts.py                   # 中文字体配置
├── output/                        # 输出文件
│   ├── full_graph/                # 完整图分析结果（4张图）
│   │   ├── full_degree_distribution.png
│   │   ├── full_clustering_coefficient.png
│   │   ├── full_path_length.png
│   │   └── full_connected_components.png
│   └── subgraph/                  # 子图分析结果（4张图）
│       ├── subgraph_acompany_with_degree_distribution.png
│       ├── subgraph_acompany_with_clustering_coefficient.png
│       ├── subgraph_acompany_with_path_length.png
│       └── subgraph_acompany_with_connected_components.png
├── reports/                       # 分析报告
│   └── medical_graph_complete_analysis.md
└── README.md                      # 项目说明文档
```

## 数据格式

### nodes.csv
```csv
node_id,name,type,other_attrs
0,安徽永生堂辛伐他汀片,Producer,{}
4,肺泡蛋白质沉积症,Disease,{"cause": "...", "desc": "...", ...}
```

**字段说明:**
- `node_id`: 节点唯一标识
- `name`: 节点名称
- `type`: 节点类型 (Disease/Drug/Food/Check/Department/Symptom/Producer)
- `other_attrs`: JSON格式的附加属性(仅Disease节点有值)

### edges.csv
```csv
src_id,dst_id,rel_type,other_attrs
4682,16526,recommand_eat,{"name": "推荐食谱"}
```

**字段说明:**
- `src_id`: 源节点ID
- `dst_id`: 目标节点ID
- `rel_type`: 关系类型
- `other_attrs`: 关系的中文描述(JSON格式)

## 关系类型

医疗知识图谱包含10种关系类型:

1. **recommand_eat** - 疾病→推荐吃食物
2. **no_eat** - 疾病→忌吃食物
3. **do_eat** - 疾病→宜吃食物
4. **common_drug** - 疾病→通用药品
5. **recommand_drug** - 疾病→推荐药品
6. **need_check** - 疾病→需要检查
7. **has_symptom** - 疾病→症状
8. **acompany_with** - 疾病→并发疾病
9. **belongs_to** - 疾病→所属科室 / 科室→科室
10. **drugs_of** - 厂商→生产药品

## 🚀 快速开始

### 1. 环境准备

#### 如果没有 .venv 虚拟环境

如果项目目录中没有 `.venv` 文件夹，请按以下步骤初始化虚拟环境：

```bash
# 1. 确保已安装 Python 3.7+
python3 --version

# 2. 创建虚拟环境
python3 -m venv .venv

# 3. 激活虚拟环境
source .venv/bin/activate

# 4. 升级 pip
pip install --upgrade pip

# 5. 安装项目依赖
pip install -r requirements.txt
```

#### 如果已有 .venv 虚拟环境

项目已配置虚拟环境，请按以下步骤设置：

```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装依赖（已预装，可选）
pip install -r requirements.txt
```

### 2. 运行分析

```bash
# 方法1: 使用便捷脚本（推荐）
./run_analysis.sh

# 方法2: 手动激活环境后运行
source .venv/bin/activate
python src/graph_analysis.py

# 方法3: 作为模块运行
source .venv/bin/activate
python3 -c "
import sys
sys.path.insert(0, '.')
from src import graph_analysis as ga
analyzer = ga.MedicalGraphAnalyzer('data/nodes.csv', 'data/edges.csv')
analyzer.load_data()
analyzer.analyze_full_graph()
analyzer.analyze_subgraphs_by_relation()
"
```

### 3. 查看结果

分析完成后会生成：

- **完整图分析**：`output/full_graph/` 目录下的 4 张图表
- **子图分析**：`output/subgraph/` 目录下的 4 张图表（仅 acompany_with 关系）
- **详细报告**：`reports/medical_graph_complete_analysis.md`

### 4. 自定义分析

如需分析其他关系类型，修改 `src/graph_analysis.py` 中的 `analyze_subgraphs_by_relation` 方法：

```python
def analyze_subgraphs_by_relation(self):
    # 修改这里，添加你想分析的关系类型
    rel_types = ['acompany_with', 'recommand_drug', 'has_symptom']
    # ...
```

## 分析指标说明

### 度分布 (Degree Distribution)

**含义**: 统计不同度数的节点数量分布

**图表特征**:
- 横轴: 度 k (对数刻度)
- 纵轴: 节点数量 P(k) (对数刻度)
- 拟合曲线: 幂律函数 P(k) ~ k^(-γ)

**医学解释**:
- γ ≈ 1.5-2.5 → 典型无标度网络
- 高度节点 → "中心疾病/症状/药物"
- 低度节点 → 专科或罕见实体

### 聚类系数 (Clustering Coefficient)

**含义**: 衡量节点的邻居之间相互连接的程度

**图表特征**:
- 横轴: 度 k (对数刻度)
- 纵轴: 平均聚类系数 C(k) (对数刻度)
- 拟合曲线: C(k) ~ k^(-β)

**医学解释**:
- 高聚类系数 → 疾病/症状形成紧密簇群
- 负幂律相关 → 层次化网络结构
- 并发关系聚类高 → 相关疾病成组出现

### 路径长度 (Path Length)

**含义**: 任意两个节点之间的最短路径长度分布

**图表特征**:
- 横轴: 距离 d
- 纵轴: 路径数量 (对数刻度)
- 峰值标注

**医学解释**:
- 平均路径长度 < 6 → 小世界网络
- 快速导航: 疾病→症状→诊断→治疗
- 峰值位置: 典型诊疗链长度

### 连通分量 (Connected Components)

**含义**: 图中相互连接的子图分布

**图表特征**:
- 横轴: 分量大小 (对数刻度)
- 纵轴: 分量数量 (对数刻度)

**医学解释**:
- 1个大分量 → 知识图谱完整连通
- 多个小分量 → 独立的医学领域/子系统
- 分量占比 → 知识覆盖程度

## 📈 分析结果亮点

### 完整图统计

- ✅ **44,110个节点**, 276,586条边
- ✅ **100%连通**, 所有节点在同一弱连通分量中
- ✅ **无标度网络** (幂律指数 γ=1.53, R²=0.800)
- ✅ **小世界特性** (平均路径长度 4.42，最大路径 8)
- ✅ 存在度为 2,755 的超级中心节点

### 并发疾病子图 (acompany_with)

- 📊 **8,572个节点**, 12,029条边
- 📊 **平均度 2.80**: 每个疾病平均并发 2-3 种其他疾病
- 📊 **聚类系数 0.0098**: 略高于完整图，仍为星型结构
- 📊 **路径长度 5.59**: 并发链略长，可预测远期风险
- 📊 **最大强连通分量 120节点**: 可能代表代谢综合征疾病群

### 关键洞察

1. **医疗知识图谱呈"以疾病为中心"的星型结构** - 低聚类系数体现
2. **具备小世界特性** - 平均4.42步可达任意实体，支持快速推理
3. **存在超级节点Hub** - 少数核心疾病拥有大量连接
4. **关系具有明确方向性** - 强连通覆盖率仅0.27%，适合因果推理
5. **并发疾病网络揭示共病模式** - 可用于疾病风险预测

## 🔍 分析指标说明

### 度分布 (Degree Distribution)

**含义**: 统计不同度数的节点数量分布，揭示网络的连接模式

**医学解释**:
- **平均度 12.54**: 每个医疗实体平均关联约13个其他实体
- **幂律指数 γ≈1.53**: 符合无标度网络，少数核心实体拥有大量连接
- **最大度 2,755**: 存在"超级节点"（如常见疾病），处于知识体系核心

### 聚类系数 (Clustering Coefficient)

**含义**: 衡量节点的邻居之间相互连接的程度，反映网络的聚集特性

**医学解释**:
- **平均聚类系数 0.0062**: 医疗实体的"朋友"之间很少直接连接
- **星型结构**: 疾病作为中心，连接症状、药物等周边实体
- **与MSN对比**: 社交网络聚类系数0.11（朋友的朋友也是朋友），医疗网络更分散

### 路径长度 (Path Length)

**含义**: 任意两个节点之间的最短路径长度分布

**医学解释**:
- **平均路径 4.42**: 任意两个实体平均通过4-5步建立关联
- **最大路径 8**: 最远的两个实体也只需8步即可连接
- **小世界网络**: 信息传递效率高，适合智能问诊和知识推理

### 连通分量 (Connected Components)

**含义**: 图中相互连接的子图分布

**医学解释**:
- **强连通分量覆盖 0.27%**: 医疗关系具有方向性（疾病→症状，不可逆）
- **弱连通分量 100%**: 忽略方向后，所有实体完全连通
- **与MSN对比**: 社交网络强连通覆盖99%（双向关系），医疗网络为单向因果关系

## ⚙️ 技术细节

### 算法复杂度

- **度分布**: O(V + E)
- **聚类系数**: O(V × d²)，其中 d 为平均度
- **路径长度**: O(S × V²)，其中 S 为采样数
- **连通分量**: O(V + E)

### 优化策略

- **大图采样**: 路径长度采样 10,000 对节点（完整图）、5,000 对节点（子图）
- **子图过滤**: 自动跳过节点数 < 10 的子图
- **规模限制**: 超过 10,000 节点的子图跳过部分耗时分析

### 拟合方法

- **幂律拟合**: 在对数空间进行线性回归
- **R² 评估**: 衡量拟合质量（> 0.7 为良好）
- **异常值处理**: 过滤度=0 或系数=0 的点

## 🔧 扩展功能建议

可以在此基础上扩展以下功能：

1. **PageRank 分析** - 找出最重要的疾病/药物
2. **HITS 算法** - 识别枢纽节点和权威节点
3. **社区发现** - Louvain 算法划分疾病社区
4. **Node2Vec** - 节点嵌入和相似性分析
5. **路径推荐** - 诊疗路径智能推荐
6. **知识补全** - 基于图结构预测缺失关系

## 🐛 故障排除

### 问题1: 中文显示为方框

**解决方案**: 检查 `src/fonts.py` 是否正确配置中文字体

### 问题2: 内存不足

**解决方案**: 减少采样数量，或只分析部分子图

### 问题3: 图像无法保存

**解决方案**: 确保 `output/` 目录有写入权限

### 问题4: 拟合失败

**原因**: 数据点太少或不符合幂律分布  
**处理**: 程序会自动捕获异常并继续执行

### 问题5: Unicode 负号警告

**已修复**: 全局设置 ASCII 负号格式化器，不再出现字体警告

## 📚 技术栈

本工具使用以下开源库：

- **NetworkX** - 复杂网络分析
- **Matplotlib** - 数据可视化
- **Pandas** - 数据处理
- **NumPy & SciPy** - 科学计算

## 📝 许可证

MIT License

## 👥 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，欢迎提 Issue！

---

**版本**: 2.0.0  
**更新日期**: 2025-11-07  
**主要更新**:
- ✅ 全局修复 Unicode 负号显示问题
- ✅ 统一图像保存路径到 `output/` 目录
- ✅ 限定子图分析为 `acompany_with` 关系
- ✅ 生成详细的完整分析报告
- ✅ 优化代码结构和文档

**作者**: Medical Knowledge Graph Analysis Team

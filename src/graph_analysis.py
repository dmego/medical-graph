"""
医疗知识图谱基本属性分析

分析内容：
1. 度分布 (Degree Distribution)
2. 路径长度 (Path Length Distribution)
3. 聚类系数 (Clustering Coefficient)
4. 连通分量 (Connected Components)
"""

import os
import sys
import warnings
from collections import Counter
from typing import Dict, List, Any, Optional, cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import linregress

# 忽略警告信息
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入数据加载器和中文图表配置
from src.data_loader import load_nodes, load_edges, OUTPUT_DIR
from src.set_chinese_plot import configure_chinese_plot_style

# 初始化matplotlib中文显示设置
configure_chinese_plot_style()


class MedicalGraphAnalyzer:
    """医疗知识图谱分析器"""
    
    def __init__(self, output_dir: str = str(OUTPUT_DIR)):
        """
        初始化分析器
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir: str = output_dir
        
        # 创建图片生成基础目录
        self.base_dir = os.path.join(output_dir, 'images', 'base')
        os.makedirs(os.path.join(self.base_dir, 'full_graph'), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, 'subgraph'), exist_ok=True)
        
    def load_data(self) -> None:
        """加载图数据"""
        print("正在加载数据...")
        try:
            self.nodes_df = load_nodes()
            self.edges_df = load_edges()
            
            if self.nodes_df is None or self.edges_df is None:
                raise ValueError("数据加载失败")
                
            print(f"节点数量: {len(self.nodes_df)}")
            print(f"边数量: {len(self.edges_df)}")
            print(f"节点类型: {self.nodes_df['type'].unique()}")
            print(f"关系类型: {self.edges_df['rel_type'].unique()}")
        except Exception as e:
            print(f"数据加载出错: {e}")
            raise
        
    def build_graph(self, rel_type: Optional[str] = None) -> nx.DiGraph:
        """
        构建图
        
        参数:
            rel_type: 关系类型，如果为None则构建完整图
            
        返回:
            构建的图对象
        """
        if self.nodes_df is None or self.edges_df is None:
            raise ValueError("数据未加载，请先调用load_data()")
            
        if rel_type is None:
            edges = self.edges_df
            print("构建完整图...")
        else:
            edges = self.edges_df[self.edges_df['rel_type'] == rel_type]
            print(f"构建关系类型为 {rel_type} 的子图...")
        
        # 创建有向图
        G: nx.DiGraph = nx.DiGraph()
        
        # 添加节点
        for _, row in self.nodes_df.iterrows():
            G.add_node(row['node_id'], 
                      name=row['name'], 
                      type=row['type'])
        
        # 添加边
        for _, row in edges.iterrows():
            G.add_edge(row['src_id'], row['dst_id'], 
                      rel_type=row['rel_type'])
        
        # 移除孤立节点
        G.remove_nodes_from(list(nx.isolates(G)))
        
        print(f"图中节点数: {G.number_of_nodes()}")
        print(f"图中边数: {G.number_of_edges()}")
        
        self.graph = G
        return G
    
    def analyze_degree_distribution(self, output_prefix: str = "full", is_subgraph: bool = False) -> Dict[str, Any]:
        """
        分析度分布
        
        参数:
            output_prefix: 输出文件前缀
            is_subgraph: 是否为子图分析
            
        返回:
            统计数据字典
        """
        if self.graph is None:
            raise ValueError("图未构建，请先调用build_graph()")
            
        print("\n=== 分析度分布 ===")
        
        # 转换为无向图来计算度
        G_undirected = self.graph.to_undirected()
        
        # 计算度
        degrees = dict(G_undirected.degree())  # type: ignore
        degree_values = list(degrees.values())
        
        # 统计度分布
        degree_counts = Counter(degree_values)
        degrees_sorted = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees_sorted]
        
        print(f"最小度: {min(degree_values)}")
        print(f"最大度: {max(degree_values)}")
        print(f"平均度: {np.mean(degree_values):.2f}")
        print(f"度的中位数: {np.median(degree_values):.2f}")
        
        # 确定输出路径
        subdir = 'subgraph' if is_subgraph else 'full_graph'
        output_path = os.path.join(self.base_dir, subdir, f'{output_prefix}_degree_distribution.png')
        
        # 幂律拟合 (只对度>0的数据拟合，用于输出统计信息)
        valid_idx = [i for i, d in enumerate(degrees_sorted) if d > 0 and counts[i] > 0]
        x_fit = np.array([degrees_sorted[i] for i in valid_idx])
        y_fit = np.array([counts[i] for i in valid_idx])
        # 为拟合结果设置默认值，避免未定义变量
        has_fit = False
        gamma = None
        r_value = 0

        try:
            # 在对数空间进行线性拟合
            log_x = np.log10(x_fit)
            log_y = np.log10(y_fit)
            slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)  # type: ignore

            gamma = -slope  # type: ignore
            has_fit = True
            print(f"幂律指数 γ = {gamma:.2f}")
            print(f"R² = {r_value**2:.3f}")  # type: ignore
        except Exception as e:
            print(f"拟合失败: {e}")
        
        # 绘制双对数度分布图
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 散点图 - 使用酒红色空心圆圈,点更小
        ax.scatter(degrees_sorted, counts, alpha=0.8, s=25, 
                  facecolors='none', edgecolors='#8B0000', linewidth=1.2)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # 设置y轴范围，向下扩展一点避免显示不完全
        y_min = min(counts) * 0.5  # 比最小值再小一点
        y_max = max(counts) * 2    # 比最大值再大一点
        ax.set_ylim(bottom=y_min, top=y_max)
        
        ax.set_xlabel('Degree k(度)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of nodes P(k)(节点数量)', fontsize=14, fontweight='bold')
        ax.set_title('Degree Distribution(度分布)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"度分布图已保存: {output_path}")
        
        # 返回统计数据
        return {
            'mean_degree': np.mean(degree_values),
            'median_degree': np.median(degree_values),
            'min_degree': min(degree_values),
            'max_degree': max(degree_values),
            'gamma': gamma if has_fit else None,
            'r_squared': r_value**2 if has_fit else None  # type: ignore
        }
        
    def analyze_clustering_coefficient(self, output_prefix: str = "full", is_subgraph: bool = False) -> None:
        """
        分析聚类系数
        
        参数:
            output_prefix: 输出文件前缀
            is_subgraph: 是否为子图分析
        """
        if self.graph is None:
            raise ValueError("图未构建，请先调用build_graph()")
            
        print("\n=== 分析聚类系数 ===")
        
        # 转换为无向图
        G_undirected = self.graph.to_undirected()
        
        # 计算聚类系数
        clustering_coeffs = cast(Dict[Any, float], nx.clustering(G_undirected))
        avg_clustering = nx.average_clustering(G_undirected)
        
        print(f"平均聚类系数: {avg_clustering:.4f}")
        
        # 按度汇总平均聚类系数
        degrees = dict(G_undirected.degree())  # type: ignore
        
        # 创建度到聚类系数的映射
        degree_to_clustering: Dict[int, List[float]] = {}
        for node in G_undirected.nodes():
            d = degrees[node]
            c = clustering_coeffs[node]
            if d not in degree_to_clustering:
                degree_to_clustering[d] = []
            degree_to_clustering[d].append(c)
        
        # 计算每个度的平均聚类系数
        avg_clustering_by_degree = {}
        for d, c_list in degree_to_clustering.items():
            avg_clustering_by_degree[d] = np.mean(c_list)
        
        # 排序
        degrees_sorted = sorted(avg_clustering_by_degree.keys())
        avg_clustering_values = [avg_clustering_by_degree[d] for d in degrees_sorted]
        
        # 幂律拟合 (过滤掉c=0的点，用于输出统计信息和绘制拟合线)
        valid_data = [(d, c) for d, c in zip(degrees_sorted, avg_clustering_values) 
                     if d > 1 and c > 0]
        
        has_fit = False
        slope = 0
        r_value = 0
        x_line: np.ndarray = np.array([])
        y_line: np.ndarray = np.array([])
        
        if len(valid_data) > 5:
            x_fit = np.array([d for d, c in valid_data])
            y_fit = np.array([c for d, c in valid_data])
            
            try:
                # 在对数空间进行线性拟合
                log_x = np.log10(x_fit)
                log_y = np.log10(y_fit)
                slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)  # type: ignore
                
                print(f"聚类系数幂律指数 β = {-slope:.2f}")  # type: ignore
                print(f"R² = {r_value**2:.3f}")  # type: ignore
                
                # 计算拟合线数据
                has_fit = True
                x_line = np.logspace(np.log10(min(x_fit)), np.log10(max(x_fit)), 100)
                y_line = 10**intercept * x_line**slope  # type: ignore
            except Exception as e:
                print(f"拟合失败: {e}")
                has_fit = False
        
        # 绘制聚类系数图
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 散点图 - 使用酒红色空心圆,点更小
        ax.scatter(degrees_sorted, avg_clustering_values, alpha=0.8, s=25, 
                  facecolors='none', edgecolors='#8B0000', linewidth=1.2)
        
        # 绘制拟合线
        if has_fit:
            # 使用英文减号避免字体问题
            beta_value = -slope  # type: ignore
            ax.plot(x_line, y_line, 'b--', linewidth=2, alpha=0.7,
                   label=f'Power-law fit(幂律拟合): β={beta_value:.2f}')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_ylabel('Average Clustering Coefficient C(k)(平均聚类系数)', 
                     fontsize=14, fontweight='bold')
        ax.set_title('Clustering Coefficient(聚类系数)', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if has_fit:
            ax.legend(fontsize=12)
        
        # 确定输出路径
        subdir = 'subgraph' if is_subgraph else 'full_graph'
        output_path = os.path.join(self.base_dir, subdir, f'{output_prefix}_clustering_coefficient.png')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"聚类系数图已保存: {output_path}")
        
    def analyze_path_length(self, output_prefix: str = "full", sample_size: int = 10000, is_subgraph: bool = False) -> None:
        """
        分析路径长度分布
        
        参数:
            output_prefix: 输出文件前缀
            sample_size: 采样数量
            is_subgraph: 是否为子图分析
        """
        if self.graph is None:
            raise ValueError("图未构建，请先调用build_graph()")
            
        print("\n=== 分析路径长度分布 ===")
        
        # 转换为无向图
        G_undirected = self.graph.to_undirected()
        
        # 找到最大连通分量 (WCC)
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_wcc = G_undirected.subgraph(largest_cc).copy()
        
        print(f"最大连通分量节点数: {G_wcc.number_of_nodes()}")
        print(f"最大连通分量边数: {G_wcc.number_of_edges()}")
        
        # 采样计算路径长度分布
        print(f"采样 {sample_size} 对节点计算最短路径...")
        nodes = list(G_wcc.nodes())
        path_lengths: List[int] = []
        
        np.random.seed(42)
        for _ in range(sample_size):
            src = np.random.choice(nodes)
            dst = np.random.choice(nodes)
            if src != dst:
                try:
                    length = nx.shortest_path_length(G_wcc, src, dst)
                    path_lengths.append(length)
                except nx.NetworkXNoPath:
                    pass
        
        # 统计路径长度分布
        path_counts = Counter(path_lengths)
        distances = sorted(path_counts.keys())
        counts = [path_counts[d] for d in distances]
        
        print(f"采样得到的路径数: {len(path_lengths)}")
        print(f"最短路径长度范围: {min(distances)} - {max(distances)}")
        print(f"平均路径长度 (采样): {np.mean(path_lengths):.2f}")
        
        # 绘制路径长度分布图
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 用线连接的散点图 - 使用酒红色空心圆
        ax.plot(distances, counts, '-', color='#8B0000', linewidth=2, alpha=0.8)
        ax.scatter(distances, counts, s=20, facecolors='none', 
                  edgecolors='#8B0000', linewidths=1.2, alpha=0.8)
        
        ax.set_xlabel('Path Length(路径长度)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Paths(路径数量)', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.set_title('Path Length Distribution(路径长度分布)', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 确定输出路径
        subdir = 'subgraph' if is_subgraph else 'full_graph'
        output_path = os.path.join(self.base_dir, subdir, f'{output_prefix}_path_length.png')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"路径长度分布图已保存: {output_path}")
        
    def analyze_connected_components(self, output_prefix: str = "full", is_subgraph: bool = False) -> None:
        """
        分析连通分量 (使用强连通分量)
        
        参数:
            output_prefix: 输出文件前缀
            is_subgraph: 是否为子图分析
        """
        if self.graph is None:
            raise ValueError("图未构建，请先调用build_graph()")
            
        print("\n=== 分析连通分量 ===")
        
        # 使用有向图的强连通分量
        print("使用强连通分量分析...")
        
        # 找到所有强连通分量
        sccs = list(nx.strongly_connected_components(self.graph))
        
        print(f"强连通分量数量: {len(sccs)}")
        
        # 统计各强连通分量的大小
        scc_sizes = [len(scc) for scc in sccs]
        scc_sizes.sort(reverse=True)
        
        print(f"最大强连通分量大小: {scc_sizes[0]}")
        print(f"最大强连通分量占比: {scc_sizes[0]/self.graph.number_of_nodes()*100:.2f}%")
        
        # 统计强连通分量大小分布
        size_counts = Counter(scc_sizes)
        sizes_sorted = sorted(size_counts.keys())
        counts = [size_counts[s] for s in sizes_sorted]
        
        # 绘制强连通分量分布图
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # 散点图 - 使用酒红色空心圆,点更小
        ax.scatter(sizes_sorted, counts, alpha=0.8, s=25, 
                  facecolors='none', edgecolors='#8B0000', linewidth=1.2)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Component Size(连通分量大小)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Components(连通分量数量)', fontsize=14, fontweight='bold')
        ax.set_title('Strongly Connected Components Distribution(强连通分量分布)', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 确定输出路径
        subdir = 'subgraph' if is_subgraph else 'full_graph'
        output_path = os.path.join(self.base_dir, subdir, f'{output_prefix}_connected_components.png')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"强连通分量分布图已保存: {output_path}")
        
    def analyze_full_graph(self):
        """分析完整图"""
        print("\n" + "="*60)
        print("分析完整图")
        print("="*60)
        
        self.build_graph()
        self.analyze_degree_distribution("full")
        self.analyze_clustering_coefficient("full")
        self.analyze_path_length("full")
        self.analyze_connected_components("full")
        
    def analyze_subgraphs_by_relation(self):
        """按关系类型分析子图 - 只分析 acompany_with 关系"""
        print("\n" + "="*60)
        print("按关系类型分析子图 (仅 acompany_with)")
        print("="*60)
        
        # 只分析 acompany_with 关系类型
        rel_types = ['acompany_with']
        
        for rel_type in rel_types:
            print("\n" + "-"*60)
            print(f"关系类型: {rel_type}")
            print("-"*60)
            
            try:
                self.build_graph(rel_type)
                
                if self.graph is None:
                    print(f"构建关系类型 {rel_type} 的子图失败")
                    continue
                    
                # 跳过节点数太少的子图
                if self.graph.number_of_nodes() < 10:
                    print(f"子图节点数太少({self.graph.number_of_nodes()})，跳过分析")
                    continue
                
                prefix = f"subgraph_{rel_type}"
                self.analyze_degree_distribution(prefix, is_subgraph=True)
                self.analyze_clustering_coefficient(prefix, is_subgraph=True)
                
                # 路径长度和连通分量分析可能比较耗时
                if self.graph.number_of_nodes() < 10000:
                    self.analyze_path_length(prefix, sample_size=5000, is_subgraph=True)
                    self.analyze_connected_components(prefix, is_subgraph=True)
                else:
                    print("子图规模较大，跳过路径长度和连通分量分析")
                    
            except Exception as e:
                print(f"分析关系类型 {rel_type} 时出错: {e}")
                continue


def main():
    """主函数"""
    print("医疗知识图谱基本属性分析程序")
    print("="*60)
    
    # 初始化分析器
    analyzer = MedicalGraphAnalyzer()
    
    # 加载数据
    analyzer.load_data()
    
    # 分析完整图
    analyzer.analyze_full_graph()
    
    # 分析子图
    analyzer.analyze_subgraphs_by_relation()
    
    print("\n" + "="*60)
    print("所有分析完成!")
    print("="*60)


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
@Time: 3/27/2025 8:11 PM
@Author: Honggang Yuan
@Email: honggang.yuan@nokia-sbell.com
Description: 
"""

import numpy as np
import scipy.spatial.distance as dist
import gudhi as gd
import matplotlib.pyplot as plt
from persim import plot_diagrams
from ripser import ripser

from ripser import Rips

x = np.linspace(0, 25, 1)
print(x)
rips = Rips(maxdim=2)

def compute_min_radius_vr(vectors):
    """
    计算基于 Vietoris-Rips 复形，使所有向量最终合并为一个连通分量的最小半径 r_f。
    """
    # 计算欧几里得距离矩阵
    distance_matrix = dist.pdist(vectors, metric='euclidean')

    # 转换为方阵形式
    distance_matrix = dist.squareform(distance_matrix)

    # 构造 VR 复形
    rips_complex = gd.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)

    # 获取所有 1-维单纯形（边）的死亡时间，即当它们形成连通分量时的最大半径
    edges = [simplex for simplex in simplex_tree.get_filtration() if len(simplex[0]) == 2]
    # 找到形成单一连通分量的最小半径 r_f
    r_f = max(edge[1] for edge in edges)
    return r_f


points = np.random.rand(12, 768)  # 12 个 768 维向量

distance_matrix = dist.pdist(points, metric='euclidean')
print(f"distance_matrix : {distance_matrix}")
distance_matrix = dist.squareform(distance_matrix)
# 构造 VR 复形
rips_complex = gd.RipsComplex(distance_matrix=distance_matrix)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)

# 获取所有 1-维单纯形（边）的死亡时间，即当它们形成连通分量时的最大半径
edges = [simplex for simplex in simplex_tree.get_filtration() if len(simplex[0]) == 2]
homology = np.array([(0, edge[1]) for edge in edges])
# 找到形成单一连通分量的最小半径 r_f
r_f = max(edge[1] for edge in edges)

print(f"Vietoris-Rips 复形计算的最小连通半径 r_f: {r_f:.4f}")

# 绘制 Birth-Death 持续同调图
# plt.figure(figsize=(8, 6))
# plot_diagrams([homology], show=True, title="0-Persistent Homology Birth-Death Diagram")
# plt.axhline(y=r_f, color='r', linestyle='--', label=f"r_f = {r_f:.4f}")
# plt.legend()
# plt.show()

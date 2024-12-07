import os
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import pickle
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
import numpy as np
# 设置数据文件夹路径
folder_path = './output'

save_path = "./TME"
# 获取文件夹中所有 .nuclei.csv 文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.nuclei.csv')]

# 定义 k 值
k = 8

# # 遍历每个 .nuclei.csv 文件
# for csv_file in csv_files:
#     pkl_file1 = os.path.join(save_path, csv_file.replace('.nuclei.csv', '_graph.pkl'))
#
#     # 检查 .pkl 文件是否已经存在，存在则跳过
#     if os.path.exists(pkl_file1):
#         print(f"Skipping {csv_file}, corresponding .pkl file already exists.")
#         continue  # 跳过已处理的文件
#     # 读取细胞数据
#     file_path = os.path.join(folder_path, csv_file)
#     df = pd.read_csv(file_path)
#
#     # 提取中心点坐标 (x_c, y_c)
#     coordinates = df[['x_c', 'y_c']].values
#     # 提取需要的特征：labels, scores, box_area
#     labels = df['labels'].values  # 假设 'labels' 列名为 'labels'
#     # 将 -100 替换为 7
#     labels = np.where(labels == -100, 8, labels)
#     scores = df['scores'].values  # 假设 'scores' 列名为 'scores'
#     box_area = df['box_area'].values  # 假设 'box_area' 列名为 'box_area'
#
#     # 对 box_area 进行归一化处理
#     scaler = MinMaxScaler()
#     box_area_normalized = scaler.fit_transform(box_area.reshape(-1, 1)).flatten()
#
#     # 使用 KNN 进行邻居搜索
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coordinates)
#     distances, indices = nbrs.kneighbors(coordinates)
#
#     # 获取 unique 的标签数目，作为 one-hot 编码的基础
#     num_classes = len(set(labels))
#
#     # 构建一个无向图，节点是细胞，边是邻居关系
#     G = nx.Graph()
#
#     # 添加细胞节点
#     for i in range(len(coordinates)):
#         # labels = labels.tolist()
#         # 将 labels 进行 one-hot 编码
#         label_onehot = F.one_hot(torch.tensor(labels[i] - 1), num_classes=9).numpy()
#         node_features = {
#             'pos': (coordinates[i][0], coordinates[i][1]),  # 位置
#             'label': labels[i],  # 原始标签
#             'score': scores[i],  # 细胞核的得分
#             'box_area': box_area_normalized[i],  # 归一化后的细胞核面积
#             'features': list(label_onehot) + [scores[i], box_area_normalized[i]]  # 特征拼接
#         }
#         G.add_node(i, **node_features)
#
#     # 添加邻居关系作为边
#     for i, neighbors in enumerate(indices):
#         for neighbor in neighbors[1:]:  # 跳过自己
#             G.add_edge(i, neighbor)
#
#     from torch_geometric.utils import from_networkx
#
#     bag_feats_TME = from_networkx(G)
#     bag_feats_TME.node = bag_feats_TME.label.view(-1, 1).float()  # 将 label 作为节点特征
#     # 生成 .pkl 文件名，以文件名为基础
#     pkl_file = os.path.join(save_path, csv_file.replace('.nuclei.csv', '_graph.pkl'))
#
#     # 将图保存为 .pkl 文件
#     with open(pkl_file, 'wb') as f:
#         pickle.dump(bag_feats_TME, f)
#
#     print(f"Graph for {csv_file} saved to {pkl_file}")


import h5py

# 遍历每个 .nuclei.csv 文件
for csv_file in csv_files:
    h5_file = os.path.join(save_path, csv_file.replace('.nuclei.csv', '_graph.h5'))

    # 检查 .h5 文件是否已经存在，存在则跳过
    if os.path.exists(h5_file):
        print(f"Skipping {csv_file}, corresponding .h5 file already exists.")
        continue  # 跳过已处理的文件
    # 读取细胞数据
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)

    # 提取中心点坐标 (x_c, y_c)
    coordinates = df[['x_c', 'y_c']].values
    # 提取需要的特征：labels, scores, box_area
    labels = df['labels'].values  # 假设 'labels' 列名为 'labels'
    # 将 -100 替换为 7
    labels = np.where(labels == -100, 8, labels)
    scores = df['scores'].values  # 假设 'scores' 列名为 'scores'
    box_area = df['box_area'].values  # 假设 'box_area' 列名为 'box_area'

    # 对 box_area 进行归一化处理
    scaler = MinMaxScaler()
    box_area_normalized = scaler.fit_transform(box_area.reshape(-1, 1)).flatten()

    # 使用 KNN 进行邻居搜索
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coordinates)
    distances, indices = nbrs.kneighbors(coordinates)

    # 获取 unique 的标签数目，作为 one-hot 编码的基础
    num_classes = len(set(labels))

    # 构建一个无向图，节点是细胞，边是邻居关系
    G = nx.Graph()

    # 添加细胞节点
    for i in range(len(coordinates)):
        # 将 labels 进行 one-hot 编码
        label_onehot = F.one_hot(torch.tensor(labels[i] - 1), num_classes=9).numpy()
        node_features = {
            'pos': (coordinates[i][0], coordinates[i][1]),  # 位置
            'label': labels[i],  # 原始标签
            'score': scores[i],  # 细胞核的得分
            'box_area': box_area_normalized[i],  # 归一化后的细胞核面积
            'features': list(label_onehot) + [scores[i], box_area_normalized[i]]  # 特征拼接
        }
        G.add_node(i, **node_features)

    # 添加邻居关系作为边
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  # 跳过自己
            G.add_edge(i, neighbor)

    from torch_geometric.utils import from_networkx

    bag_feats_TME = from_networkx(G)
    # bag_feats_TME.node = bag_feats_TME.label.view(-1, 1).float()  # 将 label 作为节点特征
    a = bag_feats_TME.features.numpy()

    # 保存图数据到 .h5 文件
    with h5py.File(h5_file, 'w') as hf:
        hf.create_dataset('node_features', data=bag_feats_TME.features.numpy())
        hf.create_dataset('edges', data=bag_feats_TME.edge_index.numpy())
        hf.create_dataset('labels', data=bag_feats_TME.label.numpy())
        # hf.create_dataset('graph', data=bag_feats_TME)

    print(f"Graph for {csv_file} saved to {h5_file}")

# import os
# import pandas as pd
# import pickle
# import networkx as nx
# from sklearn.neighbors import NearestNeighbors
#
# # 设置数据文件夹路径
# folder_path = 'M:\\project_P53\\hd_wsi-master\\test_features'
# save_path = "M:\\project_P53\\lung_cancer\\TME"
#
# # 获取文件夹中所有 .nuclei.csv 文件
# csv_files = [f for f in os.listdir(folder_path) if f.endswith('.nuclei.csv')]
#
# # 定义 k 值
# k = 8
#
# # 遍历每个 .nuclei.csv 文件
# for csv_file in csv_files:
#     pkl_file1 = os.path.join(save_path, csv_file.replace('.nuclei.csv', '_graph.pkl'))
#
#     # 检查 .pkl 文件是否已经存在，存在则跳过
#     if os.path.exists(pkl_file1):
#         print(f"Skipping {csv_file}, corresponding .pkl file already exists.")
#         continue  # 跳过已处理的文件
#
#     # 读取细胞数据
#     file_path = os.path.join(folder_path, csv_file)
#     df = pd.read_csv(file_path)
#
#     # 提取中心点坐标 (x_c, y_c)
#     coordinates = df[['x_c', 'y_c']].values
#
#     # 使用 KNN 进行邻居搜索
#     nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coordinates)
#     distances, indices = nbrs.kneighbors(coordinates)
#
#     # 构建一个无向图，节点是细胞，边是邻居关系
#     G = nx.Graph()
#
#     # 添加细胞节点
#     for i in range(len(coordinates)):
#         G.add_node(i, pos=(coordinates[i][0], coordinates[i][1]), label=df['labels'][i])
#
#     # 添加邻居关系作为边
#     for i, neighbors in enumerate(indices):
#         for neighbor in neighbors[1:]:  # 跳过自己
#             G.add_edge(i, neighbor)
#
#     # 根据label构建不同的子图
#     subgraphs = {}
#     unique_labels = df['labels'].unique()
#
#     for label in unique_labels:
#         # 创建子图：只包含特定label的细胞及其邻居
#         subgraph_nodes = [i for i in G.nodes() if G.nodes[i]['label'] == label]
#         subgraph = G.subgraph(subgraph_nodes).copy()
#         subgraphs[label] = subgraph
#
#     # 生成 .pkl 文件名，以文件名为基础
#     pkl_file = os.path.join(save_path, csv_file.replace('.nuclei.csv', '_graph.pkl'))
#
#     # 将整个图和子图一起保存为 .pkl 文件
#     with open(pkl_file, 'wb') as f:
#         pickle.dump({'full_graph': G, 'subgraphs': subgraphs}, f)
#
#     print(f"Graph and subgraphs for {csv_file} saved to {pkl_file}")

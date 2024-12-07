#单独为每个文件构图
# from dataloader import get_edge_index_image
import torch
import os
import dgl
from PIL import Image
import torch
import joblib
import numpy as np
from torch_geometric.data import Data

import torch
import numpy as np
import joblib
import nmslib
from itertools import chain
import dgl
from scipy.stats import pearsonr
def get_edge_index_image(t_img_fea):
    start = []
    end = []
    # if id in t_img_fea:
    patch_id = {}
    i=0
    for x in t_img_fea:
        patch_id[x.split('.')[0]] = i
        i+=1
#     print(patch_id)
    for x in patch_id:
#         print(x)
        i = int(x.split('_')[0])
        j = int(x.split('_')[1])#.split('-')[1])
        # j = int(x.split('.')[0].split('-')[1])
        if str(i)+'_'+str(j+1) in patch_id:
            start.append(patch_id[str(i)+'_'+str(j)])
            end.append(patch_id[str(i)+'_'+str(j+1)])
        if str(i)+'_'+str(j-1) in patch_id:
            start.append(patch_id[str(i)+'_'+str(j)])
            end.append(patch_id[str(i)+'_'+str(j-1)])
        if str(i+1)+'_'+str(j) in patch_id:
            start.append(patch_id[str(i)+'_'+str(j)])
            end.append(patch_id[str(i+1)+'_'+str(j)])
        if str(i-1)+'_'+str(j) in patch_id:
            start.append(patch_id[str(i)+'_'+str(j)])
            end.append(patch_id[str(i-1)+'_'+str(j)])
        if str(i+1)+'_'+str(j+1) in patch_id:
            start.append(patch_id[str(i)+'_'+str(j)])
            end.append(patch_id[str(i+1)+'_'+str(j+1)])
        if str(i-1)+'_'+str(j+1) in patch_id:
            start.append(patch_id[str(i)+'_'+str(j)])
            end.append(patch_id[str(i-1)+'_'+str(j+1)])
        if str(i+1)+'_'+str(j-1) in patch_id:
            start.append(patch_id[str(i)+'_'+str(j)])
            end.append(patch_id[str(i+1)+'_'+str(j-1)])
        if str(i-1)+'_'+str(j-1) in patch_id:
            start.append(patch_id[str(i)+'_'+str(j)])
            end.append(patch_id[str(i-1)+'_'+str(j-1)])

    return [start,end]

def get_node(t_img_fea_256, t_img_fea_512, t_img_fea_1024):
    f_img_256 = []
    f_img_512 = []
    f_img_1024 = []

    for z in t_img_fea_256:
        f_img_256.append(t_img_fea_256[z])

    for z in t_img_fea_512:
        f_img_512.append(t_img_fea_512[z])

    for z in t_img_fea_1024:
        f_img_1024.append(t_img_fea_1024[z])

    return f_img_256, f_img_512, f_img_1024
class Hnsw:
    """
    KNN model cloned from https://github.com/mahmoodlab/Patch-GCN/blob/master/WSI-Graph%20Construction.ipynb
    """

    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

knn_model = Hnsw(space='l2')
# def get_node(t_img_fea_cnn):
#     f_img_cnn = []
#
#     for z in t_img_fea_cnn:
#         f_img_cnn.append(t_img_fea_cnn[z])
#
#     return f_img_cnn
def perform_pca(data, n):
    # 假设使用PCA对数据进行降维处理
    # 这里只是一个示例，实际应用中需要使用合适的降维方法
    # 在这里，我们使用PCA将数据降维到n维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    reduced_data = pca.fit_transform(data)
    return reduced_data

folder_path_256 = './features/FEATURES_DIRECTORY_256/pt_files'#'R:\\zhengda\\np'
folder_path_512 = './features/FEATURES_DIRECTORY_512/pt_files'#'R:\\zhengda\\np_swim'
folder_path_1024 = './features/FEATURES_DIRECTORY_1024/pt_files'
# save_path_swim = 'P:\\xiangya2\\STAS\\stas_all\\graph\\swim'
# save_path_all = 'P:\\xiangya2\\STAS\\stas_all\\graph\\cnn_swim'
path = "./multi_graph_1"
all_data_cnn = {}
all_data_swim = {}
all_data = {}

# 打开文件
# with open('./val.txt', 'r') as file:
#     # 读取文件的每一行，并提取第一列数据
#     first_column_data = [line.split()[0] for line in file]

# first_column_data = joblib.load('./heatmaps.pkl')#('P:\\IBM\\PKG - CPTAC-LUAD_v12\\luad_patients.pkl')
# 读取文件夹中的所有文件名，并去掉后缀生成列表
first_column_data = [os.path.splitext(file)[0] for file in os.listdir(folder_path_1024) if os.path.isfile(os.path.join(folder_path_1024, file))]
first_column_data1 = [os.path.splitext(file)[0] for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
# 遍历文件夹中的所有文件
# for file_name in os.listdir(folder_path_256_cnn):
#     if file_name.endswith('.pkl'):
for file_name in first_column_data:
    # file_name = file_name.split('.')[0]
    if '.pt' not in file_name:
        file_name = file_name +'.pt'
    if file_name.split('.')[0] in first_column_data1:
        print(file_name.split('.')[0])
        continue
    image_path_256_fea = os.path.join(folder_path_256, file_name)
    image_path_256_fea = torch.load(image_path_256_fea)

    image_path_512_fea = os.path.join(folder_path_512, file_name)
    image_path_512_fea = torch.load(image_path_512_fea)

    image_path_1024_fea = os.path.join(folder_path_1024, file_name)
    image_path_1024_fea = torch.load(image_path_1024_fea)

    # from data.dataloader import get_node#, get_edge_index_image

    node_image_path_256_fea, node_image_path_512_fea, node_image_path_1024_fea = image_path_256_fea, image_path_512_fea, image_path_1024_fea#get_node(image_path_256_fea, image_path_512_fea, image_path_1024_fea)
    # node_image_path_256_fea = torch.Tensor(np.stack(node_image_path_256_fea))
    # node_image_path_512_fea = torch.Tensor(np.stack(node_image_path_512_fea))
    # node_image_path_1024_fea = torch.Tensor(np.stack(node_image_path_1024_fea))
    #构建256的边
    n_patches = len(image_path_256_fea)  # .shape[0]
    # 使用列表推导式提取所有值
    all_values = [value for value in image_path_256_fea]

    # 使用vstack函数将所有值堆叠在一起
    image_path_256_fea = np.vstack(all_values)
    radius = 9
    # Construct graph using spatial coordinates
    knn_model.fit(image_path_256_fea)

    a = np.repeat(range(n_patches), radius - 1)
    b = np.fromiter(
        chain(
            *[knn_model.query(image_path_256_fea[v_idx], topn=radius)[1:] for v_idx in range(n_patches)]
        ), dtype=int
    )
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

    # Create edge types
    edge_type = []
    edge_sim = []
    for (idx_a, idx_b) in zip(a, b):
        metric = pearsonr
        corr = metric(image_path_256_fea[idx_a], image_path_256_fea[idx_b])[0]
        edge_type.append(1 if corr > 0 else 0)
        edge_sim.append(corr)

    # Construct dgl heterogeneous graph
    graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
    image_path_256_fea = torch.tensor(image_path_256_fea, device='cpu').float()
    # 获取图中的边
    edges1 = graph.edges()
    edge_index_image_256 = torch.tensor(torch.stack(edges1, dim=0), dtype=torch.long)

    # 构建512的边
    n_patches = len(image_path_512_fea)  # .shape[0]
    # 使用列表推导式提取所有值
    all_values = [value for value in image_path_512_fea]

    # 使用vstack函数将所有值堆叠在一起
    image_path_512_fea = np.vstack(all_values)
    radius = 9
    # Construct graph using spatial coordinates
    knn_model.fit(image_path_512_fea)

    a = np.repeat(range(n_patches), radius - 1)
    b = np.fromiter(
        chain(
            *[knn_model.query(image_path_512_fea[v_idx], topn=radius)[1:] for v_idx in range(n_patches)]
        ), dtype=int
    )
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

    # Create edge types
    edge_type = []
    edge_sim = []
    for (idx_a, idx_b) in zip(a, b):
        metric = pearsonr
        corr = metric(image_path_512_fea[idx_a], image_path_512_fea[idx_b])[0]
        edge_type.append(1 if corr > 0 else 0)
        edge_sim.append(corr)

    # Construct dgl heterogeneous graph
    graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
    image_path_512_fea = torch.tensor(image_path_512_fea, device='cpu').float()
    # 获取图中的边
    edges1 = graph.edges()
    edge_index_image_512 = torch.tensor(torch.stack(edges1, dim=0), dtype=torch.long)

    # 构建1024的边
    n_patches = len(image_path_1024_fea)  # .shape[0]
    # 使用列表推导式提取所有值
    all_values = [value for value in image_path_1024_fea]

    # 使用vstack函数将所有值堆叠在一起
    image_path_1024_fea = np.vstack(all_values)
    radius = 9
    # Construct graph using spatial coordinates
    knn_model.fit(image_path_1024_fea)

    a = np.repeat(range(n_patches), radius - 1)
    b = np.fromiter(
        chain(
            *[knn_model.query(image_path_1024_fea[v_idx], topn=radius)[1:] for v_idx in range(n_patches)]
        ), dtype=int
    )
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

    # Create edge types
    edge_type = []
    edge_sim = []
    for (idx_a, idx_b) in zip(a, b):
        metric = pearsonr
        corr = metric(image_path_1024_fea[idx_a], image_path_1024_fea[idx_b])[0]
        edge_type.append(1 if corr > 0 else 0)
        edge_sim.append(corr)

    # Construct dgl heterogeneous graph
    graph = dgl.graph((edge_spatial[0, :], edge_spatial[1, :]))
    image_path_1024_fea = torch.tensor(image_path_1024_fea, device='cpu').float()
    # 获取图中的边
    edges1 = graph.edges()
    edge_index_image_1024 = torch.tensor(torch.stack(edges1, dim=0), dtype=torch.long)


    # data_all = Data(x_img_256=node_image_path_256_fea, x_img_256_edge = edge_index_image_256, x_img_512=node_image_path_512_fea, x_img_512_edge = edge_index_image_512, x_img_1024=node_image_path_1024_fea, x_img_1024_edge = edge_index_image_1024)#, id = file_name)
    import h5py

    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()  # 确保张量在 CPU 上
        return tensor


    h5_file_path = './multi_graph_1/{}.h5'.format(file_name.split('.')[0])
    with h5py.File(h5_file_path, 'w') as hf:
        # 保存 x_img_256
        hf.create_dataset('x_img_256', data=to_numpy(node_image_path_256_fea), compression='gzip', compression_opts=9)
        hf.create_dataset('x_img_256_edge', data=to_numpy(edge_index_image_256), compression='gzip', compression_opts=9)

        # 保存 x_img_512
        hf.create_dataset('x_img_512', data=to_numpy(node_image_path_512_fea), compression='gzip', compression_opts=9)
        hf.create_dataset('x_img_512_edge', data=to_numpy(edge_index_image_512), compression='gzip', compression_opts=9)

        # 保存 x_img_1024
        hf.create_dataset('x_img_1024', data=to_numpy(node_image_path_1024_fea), compression='gzip', compression_opts=9)
        hf.create_dataset('x_img_1024_edge', data=to_numpy(edge_index_image_1024), compression='gzip', compression_opts=9)


    # joblib.dump(data_all, './features/multi_graph_1/{}.pkl'.format(file_name.split('.')[0]))





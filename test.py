import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, f1_score,average_precision_score,precision_recall_curve

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, classification_report
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from torch.utils.data import Dataset
import redis
import pickle
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    roc_auc_score, roc_curve
import random
import torch.backends.cudnn as cudnn
import json
import joblib
torch.multiprocessing.set_sharing_strategy('file_system')
import os
from Opt.lookahead import Lookahead
from Opt.radam import RAdam
from torch.cuda.amp import GradScaler, autocast
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
import h5py

feats_TME = 'LUAD_feature/256/FEATURES_DIRECTORY/pt_files'
import os

# 读取文件夹中所有文件的路径
def list_files_in_directory(directory_path):
    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

# 示例使用
directory_path = './TME'  # 请将此路径替换为您的文件夹路径
file_paths = list_files_in_directory(directory_path)

feats_TME = file_paths[0]
with h5py.File(feats_TME, 'r') as hf:
    # 读取 node_features
    node_features = hf['node_features'][:]

    # 读取 edges
    edges = hf['edges'][:]

    # # 读取 labels
    # labels = hf['labels'][:]


feats_csv_path1 = feats_TME.replace('TME\\', 'multi_graph_1/')
feats_csv_path1 = feats_csv_path1.replace('_graph.h5', '.h5')
with h5py.File(feats_csv_path1, 'r') as hf:
    # 读取 x_img_256 和对应的 edge
    x_img_256 = hf['x_img_256'][:]
    x_img_256_edge = hf['x_img_256_edge'][:]

    # 读取 x_img_512 和对应的 edge
    x_img_512 = hf['x_img_512'][:]
    x_img_512_edge = hf['x_img_512_edge'][:]

    # 读取 x_img_1024 和对应的 edge
    x_img_1024 = hf['x_img_1024'][:]
    x_img_1024_edge = hf['x_img_1024_edge'][:]

node_image_path_256_fea = torch.Tensor(x_img_256).to(device)
node_image_path_512_fea = torch.Tensor(x_img_512).to(device)
node_image_path_1024_fea = torch.Tensor(x_img_1024).to(device)
edge_index_image_256 = torch.from_numpy(x_img_256_edge).to(device)
edge_index_image_512 = torch.from_numpy(x_img_512_edge).to(device)
edge_index_image_1024 = torch.from_numpy(x_img_1024_edge).to(device)
node_features = torch.Tensor(node_features).to(device)
edges = torch.from_numpy(edges).to(device)


####基因突变
import Models.our2 as mil
# from Models import our as mil
milnet_tp53 = mil.fusion_model_graph(in_channels=768, hidden_channels=300, out_channels=2).to(device)
model_tp53 = milnet_tp53.to(device)
model_tp53_td = torch.load(r'./test_models/TP53/2_3.pth', map_location=torch.device('cpu'))
model_tp53.load_state_dict(model_tp53_td, strict=False)
model_tp53.eval()

milnet_egfr = mil.fusion_model_graph(in_channels=768, hidden_channels=300, out_channels=2).to(device)
model_egfr = milnet_egfr.to(device)
model_egfr_td = torch.load(r'./test_models/EGFR/1_4.pth', map_location=torch.device('cpu'))
model_egfr.load_state_dict(model_egfr_td, strict=False)
model_egfr.eval()

milnet_kras = mil.fusion_model_graph(in_channels=768, hidden_channels=300, out_channels=2).to(device)
model_kras = milnet_kras.to(device)
model_kras_td = torch.load(r'./test_models/KRAS/1_4.pth', map_location=torch.device('cpu'))
model_kras.load_state_dict(model_kras_td, strict=False)
model_kras.eval()

milnet_alk = mil.fusion_model_graph(in_channels=768, hidden_channels=300, out_channels=2).to(device)
model_alk = milnet_alk.to(device)
model_alk_td = torch.load(r'./test_models/ALK/1_4.pth', map_location=torch.device('cpu'))
model_alk.load_state_dict(model_alk_td, strict=False)
model_alk.eval()

milnet_tmb = mil.fusion_model_graph(in_channels=768, hidden_channels=300, out_channels=2).to(device)
model_tmb = milnet_tmb.to(device)
model_tmb_td = torch.load(r'./test_models/TMB/1_4.pth', map_location=torch.device('cpu'))
model_tmb.load_state_dict(model_tmb_td, strict=False)
model_tmb.eval()

# 获取模型输出
_, _, _, _, results_dict_adapt_tp53 = model_tp53(node_image_path_256_fea, node_image_path_512_fea,node_image_path_1024_fea,edge_index_image_256,edge_index_image_512,edge_index_image_1024,node_features, edges)
if results_dict_adapt_tp53['Y_prob'] ==0:
    tp53_mut ='No mutation'
else:
    tp53_mut = 'Mutation'

_, _, _, _, results_dict_adapt_egfr = model_egfr(node_image_path_256_fea, node_image_path_512_fea,node_image_path_1024_fea,edge_index_image_256,edge_index_image_512,edge_index_image_1024,node_features, edges)
if results_dict_adapt_egfr['Y_prob'] ==0:
    egfr_mut ='No mutation'
else:
    egfr_mut = 'Mutation'

_, _, _, _, results_dict_adapt_kras = model_kras(node_image_path_256_fea, node_image_path_512_fea,node_image_path_1024_fea,edge_index_image_256,edge_index_image_512,edge_index_image_1024,node_features, edges)
if results_dict_adapt_kras['Y_prob'] ==0:
    kras_mut ='No mutation'
else:
    kras_mut = 'Mutation'

_, _, _, _, results_dict_adapt_alk = model_alk(node_image_path_256_fea, node_image_path_512_fea,node_image_path_1024_fea,edge_index_image_256,edge_index_image_512,edge_index_image_1024,node_features, edges)
if results_dict_adapt_alk['Y_prob'] ==0:
    alk_mut ='No mutation'
else:
    alk_mut = 'Mutation'

_, _, _, _, results_dict_adapt_tmb = model_tmb(node_image_path_256_fea, node_image_path_512_fea,node_image_path_1024_fea,edge_index_image_256,edge_index_image_512,edge_index_image_1024,node_features, edges)
if results_dict_adapt_tmb['Y_prob'] ==0:
    tmb_mut ='Low'
else:
    tmb_mut = 'High'


##########基因突变亚型
import Models.our2 as mil
# from Models import our as mil
milnet_tp53_sub = mil.fusion_model_graph(in_channels=768, hidden_channels=300, out_channels=3).to(device)
model_tp53_sub = milnet_tp53_sub.to(device)
model_tp53_td_sub = torch.load(r'./test_models/TP53_SUB/1_3.pth', map_location=torch.device('cpu'))
model_tp53_sub.load_state_dict(model_tp53_td_sub, strict=False)
model_tp53_sub.eval()

# 获取模型输出
_, _, _, _, results_dict_adapt_tp53 = model_tp53(node_image_path_256_fea, node_image_path_512_fea,node_image_path_1024_fea,edge_index_image_256,edge_index_image_512,edge_index_image_1024,node_features, edges)
if results_dict_adapt_tp53['Y_prob'] ==0:
    tp53_sub ='wild type'
elif results_dict_adapt_tp53['Y_prob'] ==1:
    tp53_sub = 'Nonsense mutation'
else:
    tp53_sub = 'missense mutation'

###基因突变外显子

import Models.our2 as mil
# from Models import our as mil
milnet_tp53_out = mil.fusion_model_graph(in_channels=768, hidden_channels=300, out_channels=5).to(device)
model_tp53_out = milnet_tp53_out.to(device)
model_tp53_td_out = torch.load(r'./test_models/TP53_OUT/1_4.pth', map_location=torch.device('cpu'))
model_tp53_out.load_state_dict(model_tp53_td_out, strict=False)
model_tp53_out.eval()

milnet_egfr_out = mil.fusion_model_graph(in_channels=768, hidden_channels=300, out_channels=3).to(device)
model_egfr_out = milnet_egfr_out.to(device)
model_egfr_td_out = torch.load(r'./test_models/EGFR_OUT/1_4.pth', map_location=torch.device('cpu'))
model_egfr_out.load_state_dict(model_egfr_td_out, strict=False)
model_egfr_out.eval()

from Models import our as mil1
milnet_kras_out = mil1.fusion_model_graph(in_channels=768, hidden_channels=300, out_channels=2).to(device)
model_kras_out = milnet_kras_out.to(device)
model_kras_td_out = torch.load(r'./test_models/KRAS_OUT/6_4.pth', map_location=torch.device('cpu'))
model_kras_out.load_state_dict(model_kras_td_out, strict=False)
model_kras_out.eval()

milnet_alk_out = mil.fusion_model_graph(in_channels=768, hidden_channels=300, out_channels=2).to(device)
model_alk_out = milnet_alk_out.to(device)
model_alk_td_out = torch.load(r'./test_models/ALK_OUT/1_4.pth', map_location=torch.device('cpu'))
model_alk_out.load_state_dict(model_alk_td_out, strict=False)
model_alk_out.eval()


# 获取模型输出
if tp53_mut =='Mutation':
    _, _, _, _, results_dict_adapt_tp53_out = model_tp53_out(node_image_path_256_fea, node_image_path_512_fea,node_image_path_1024_fea,edge_index_image_256,edge_index_image_512,edge_index_image_1024,node_features, edges)
    if results_dict_adapt_tp53_out['Y_prob'] ==0:
        tp53_out ='EX5'
        tp53_out_drug = 'Eprenetapopt'
    elif results_dict_adapt_tp53_out['Y_prob'] ==1:
        tp53_out = 'EX6'
        tp53_out_drug ='No recommendation'
    elif results_dict_adapt_tp53_out['Y_prob'] == 2:
        tp53_out = 'EX7'
        tp53_out_drug ='No recommendation'
    elif results_dict_adapt_tp53_out['Y_prob'] ==3:
        tp53_out = 'EX8'
        tp53_out_drug ='No recommendation'
    else:
        tp53_out = 'Other'
        tp53_out_drug = 'No recommendation'
else:
    tp53_out = None
    tp53_out_drug = None


if egfr_mut =='Mutation':
    _, _, _, _, results_dict_adapt_egfr_out = model_egfr_out(node_image_path_256_fea, node_image_path_512_fea,node_image_path_1024_fea,edge_index_image_256,edge_index_image_512,edge_index_image_1024,node_features, edges)
    if results_dict_adapt_egfr_out['Y_prob'] ==0:
        egfr_out ='EX19'
        egfr_out_drug = 'Aumolertinib, Osimertinib'
    if results_dict_adapt_egfr_out['Y_prob'] ==1:
        egfr_out = 'EX20'
        egfr_out_drug = 'Amivantamab, Mobocertinib'
    else:
        egfr_out = 'EX21'
        egfr_out_drug = 'Afatinib, Erlotinib, Gefitinib, Osimertinib'
else:
    egfr_out = None
    egfr_out_drug = None


if kras_mut =='Mutation':
    _, _, _, _, results_dict_adapt_kras_out = model_kras_out(node_image_path_256_fea, node_image_path_512_fea,node_image_path_1024_fea,edge_index_image_256,edge_index_image_512,edge_index_image_1024,node_features, edges)
    if results_dict_adapt_kras_out['Y_prob'] ==0:
        kras_out ='Other'
        kras_out_drug ='Adagrasib, Sotorasib'
    elif results_dict_adapt_kras_out['Y_prob'] ==1:
        kras_out = 'EX2'
        kras_out_drug = 'Adagrasib, Sotorasib, '
else:
    kras_out = None
    kras_out_drug = None

if alk_mut =='Mutation':
    _, _, _, _, results_dict_adapt_alk_out = model_alk_out(node_image_path_256_fea, node_image_path_512_fea,node_image_path_1024_fea,edge_index_image_256,edge_index_image_512,edge_index_image_1024,node_features, edges)
    if results_dict_adapt_alk_out['Y_prob'] ==0:
        alk_out ='Other'
        alk_out_drug ='No recommendation'
    else:
        alk_out = 'EML4-ALK'
        alk_out_drug = 'Alectinib,Brigatinib,Lorlatinib'
else:
    alk_out = None
    alk_out_drug = None

import pandas as pd

# # 构建包含标题的空 Excel 表格
# columns = ['TP53', 'EGFR', 'KRAS', 'ALK', 'TMB']
# index = ['State', 'Subtype', 'Exon']
# data = pd.DataFrame(index=index, columns=columns)
#
# # 将 DataFrame 写入 Excel 文件
# output_file = './gene_info.xlsx'
# data.to_excel(output_file)
#
# print(f'Excel file "{output_file}" has been created with the specified structure.')
# 初始化Excel表格结构
data = {
    '': ['State', 'Subtype', 'Exon', 'Targeted Drug Recommendation'],
    'TP53': ['', '', '', ''],
    'EGFR': ['', '', '', ''],
    'KRAS': ['', '', '', ''],
    'ALK': ['', '', '', ''],
    'TMB': ['', '', '', '']
}

# 创建DataFrame
df = pd.DataFrame(data)

# 填写State行
if tp53_mut:
    df.loc[df[''] == 'State', 'TP53'] = tp53_mut
if egfr_mut:
    df.loc[df[''] == 'State', 'EGFR'] = egfr_mut
if kras_mut:
    df.loc[df[''] == 'State', 'KRAS'] = kras_mut
if alk_mut:
    df.loc[df[''] == 'State', 'ALK'] = alk_mut
if tmb_mut:
    df.loc[df[''] == 'State', 'TMB'] = tmb_mut
# 填写Subtype行
if tp53_sub:
    df.loc[df[''] == 'Subtype', 'TP53'] = tp53_sub

# 填写Exon行
if tp53_out:
    df.loc[df[''] == 'Exon', 'TP53'] = tp53_out
if egfr_out:
    df.loc[df[''] == 'Exon', 'EGFR'] = egfr_out
if kras_out:
    df.loc[df[''] == 'Exon', 'KRAS'] = kras_out
if alk_out:
    df.loc[df[''] == 'Exon', 'ALK'] = alk_out

# 填写Targeted Drug Recommendation行
if tp53_out_drug:
    df.loc[df[''] == 'Targeted Drug Recommendation', 'TP53'] = tp53_out_drug
if egfr_out_drug:
    df.loc[df[''] == 'Targeted Drug Recommendation', 'EGFR'] = egfr_out_drug
if kras_out_drug:
    df.loc[df[''] == 'Targeted Drug Recommendation', 'KRAS'] = kras_out_drug
if alk_out_drug:
    df.loc[df[''] == 'Targeted Drug Recommendation', 'ALK'] = alk_out_drug

# 保存到Excel文件
df.to_excel('./output/genetic_analysis.xlsx', index=False)

print("Excel表格已生成：genetic_analysis.xlsx")
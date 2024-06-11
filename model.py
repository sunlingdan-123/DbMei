import datetime
import math

import networkx as nx
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F

import torch.sparse

import time
from numba import jit
import heapq
from collections import Counter

from util import *

import warnings

warnings.filterwarnings("ignore")


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


# 转换为稀疏矩阵表示
def focused_data_masks(all_topic_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    # 对于每个主题
    for j in range(len(all_topic_sessions)):
        session = np.unique(all_topic_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i])
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_topic_sessions), n_node))
    return matrix


def global_graph(p):
    all_sessions = p.train_data + p.test_data
    all_sessions_adjusted = [[item - 1 for item in session] for session in all_sessions]

    edge_counts = Counter(
        [(session[i], session[i + 1]) for session in all_sessions_adjusted for i in range(len(session) - 1)])
    G = nx.DiGraph()
    for edge, count in edge_counts.items():
        G.add_weighted_edges_from([edge + (count,)])

    # 计算特征向量中心度(eigenvector centrality)
    eigenvector_all = nx.eigenvector_centrality(G, max_iter=5000)
    return eigenvector_all


class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset

    def forward(self, adjacency, embedding):
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings)
        final = torch.stack(final)
        item_embeddings = torch.sum(final, 0) / (self.layers + 1)
        return item_embeddings


class NeighConv(Module):
    def __init__(self, layers, batch_size, emb_size):
        super(NeighConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers

    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding], 0)

        seq_h = []
        for i in torch.arange(len(session_item)):
            seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = torch.stack(seq_h)

        session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len)

        session = [session_emb_lgcn]
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)

        session = torch.stack(session)
        session_emb_lgcn = torch.sum(session, 0) / (self.layers + 1)
        return session_emb_lgcn


class NeighConv1(Module):
    def __init__(self, layers, batch_size, emb_size):
        super(NeighConv1, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers

    def forward(self, item_embedding, D, A, session_item, session_len):

        session_emb_lgcn = []
        for i, session in enumerate(session_item):
            seq_h = item_embedding[torch.tensor(session)]

            seq_h = torch.mean(seq_h, dim=0).unsqueeze(0)
            session_emb_lgcn.append(seq_h)
        session_emb_lgcn = torch.cat(session_emb_lgcn, dim=0)

        session = [session_emb_lgcn]
        A = trans_to_cuda(torch.Tensor(A))
        D = trans_to_cuda(torch.Tensor(D))
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)

        session = torch.stack(session)
        session_emb_lgcn = torch.sum(session, 0) / (self.layers + 1)  # 得到每个节点(会话)的表示
        return session_emb_lgcn


# 预测受控意图与非受控意图的概率
class ProbabilityNetwork(nn.Module):
    def __init__(self, input_size, layer):
        super(ProbabilityNetwork, self).__init__()
        self.layer_num = layer
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, 1)
        self.linear3 = nn.Linear(input_size, input_size)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.layer_num == 1:  # 融合后直接输出
            x = self.linear2(x)
        elif self.layer_num == 2:  # 融合后经过1层mlp后输出
            x = self.linear1(x)
            x = self.linear2(x)
        elif self.layer_num == 3:  # 融合后经过2层mlp后输出
            x = self.linear1(x)
            x = self.linear3(x)
            x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class DbMei(Module):
    def __init__(self, adjacency, n_node, lr, layers, l2, beta, dataset, emb_size, batch_size, probs, binary_matrix):
        super(DbMei, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.dataset = dataset

        self.probs = probs
        self.binary_matrix = binary_matrix

        # 将稀疏矩阵adjacency转换为PyTorch稀疏张量
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))

        self.adjacency = adjacency  # (n_node，n_node)
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_embedding = nn.Embedding(200, self.emb_size)

        self.HyperGraph = HyperConv(self.layers, dataset, self.emb_size)
        self.HyperGraph1 = HyperConv(self.layers, dataset, self.emb_size)
        self.LineGraph = NeighConv(self.layers, self.batch_size, self.emb_size)
        self.LineGraph1 = NeighConv1(self.layers, self.batch_size, self.emb_size)

        self.w_1 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.w_3 = nn.Linear(self.emb_size, self.emb_size)
        self.w_4 = nn.Linear(self.emb_size, self.emb_size)
        self.merge = nn.Linear(self.emb_size, self.emb_size)

        self.dropout15 = nn.Dropout(0.15)
        self.dropout30 = nn.Dropout(0.30)
        self.bn1 = torch.nn.BatchNorm1d(self.emb_size, affine=False)

        self.threshold = nn.Parameter(torch.Tensor(1))  # 定义可学习的参数，作为物品划分阈值

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def intent_prob(self, sessions, session_len, max_n_node, opt):
        item_embedding = self.embedding.weight
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding], 0)

        h = []
        if opt.dataset == "Amazon_Beauty":
            batch_session = [item for session in sessions for item in session]
            session_s = batch_session[:-1]
            session_e = batch_session[1:]

            edge_counts = Counter(zip(session_s, session_e))
            # 创建一个有向图
            G = nx.DiGraph()

            for edge, count in edge_counts.items():
                G.add_weighted_edges_from([(edge[0], edge[1], count)])

            eigenvector_all = nx.eigenvector_centrality(G, max_iter=8000)  # 特征向量中心度

        # 对当前批次的所有会话
        for i, session in enumerate(sessions):
            len = session_len[i]
            session = session[:len[0]]

            if opt.dataset == "Amazon_KDD":
                session_s = session[:-1]
                session_e = session[1:]

                edge_counts = Counter(zip(session_s, session_e))
                # 创建一个有向图
                G = nx.DiGraph()
                for edge, count in edge_counts.items():
                    G.add_weighted_edges_from([(edge[0], edge[1], count)])

                eigenvector_all = nx.eigenvector_centrality(G, max_iter=5000)

            # 得到特征向量中心度最大的节点索引id
            max_centrality_node = max(eigenvector_all, key=eigenvector_all.get)

            center_item_embedding = item_embedding[max_centrality_node]

            session_item_embeddings = item_embedding[session]

            similarity = F.cosine_similarity(center_item_embedding.unsqueeze(0), session_item_embeddings, dim=1)
            similarity = similarity.unsqueeze(0)

            pad_length = max_n_node - len

            similarity = F.pad(similarity, (0, pad_length), 'constant', 0)

            # 从binary_matrix中筛选出当前会话物品的子矩阵
            item_indices = [item.item() - 1 for item in session]
            current_items_matrix = self.binary_matrix[item_indices]

            ones_count = np.sum(current_items_matrix, axis=0)
            # 计算信息熵
            entropy = 0
            for topic_num in ones_count:
                ones_prob = topic_num / current_items_matrix.shape[1]
                if ones_prob == 0:
                    ones_prob = 1e-10  # 将概率值替换为一个非零值
                entropy += ones_prob * np.log2(ones_prob)
            entropy = -entropy

            entropy_tensor = torch.tensor(entropy).reshape(1, 1)
            entropy_tensor = trans_to_cuda(entropy_tensor)
            similarity = trans_to_cuda(similarity)
            combined_tensor = torch.cat((similarity, entropy_tensor), dim=1)

            h.append(combined_tensor)
        h = torch.cat(h, dim=0)

        return h

    def intent_prob1(self, sessions, session_len, max_n_node, opt, p):
        item_embedding = self.embedding.weight
        eigenvector_all = global_graph(p)

        h = []
        # 对当前批次的所有会话
        for i, session in enumerate(sessions):
            len = session_len[i]
            session = session[:len[0]]

            # 得到特征向量中心度最大的节点索引id
            session_eigenvector_centrality = {node: eigenvector_all[node] for node in session}
            max_centrality_node = max(session_eigenvector_centrality, key=session_eigenvector_centrality.get)

            center_item_embedding = item_embedding[max_centrality_node]
            session_item_embeddings = item_embedding[session]

            similarity = F.cosine_similarity(center_item_embedding.unsqueeze(0), session_item_embeddings, dim=1)
            similarity = similarity.unsqueeze(0)
            pad_length = max_n_node - len
            similarity = F.pad(similarity, (0, pad_length), 'constant', 0)

            # 从binary_matrix中筛选出当前会话物品的子矩阵
            item_indices = [item - 1 for item in session]
            current_items_matrix = self.binary_matrix[item_indices]
            ones_count = np.sum(current_items_matrix, axis=0)
            # 计算信息熵
            entropy = 0
            for topic_num in ones_count:
                ones_prob = topic_num / current_items_matrix.shape[1]
                if ones_prob == 0:
                    ones_prob = 1e-10
                entropy += ones_prob * np.log2(ones_prob)
            entropy = -entropy

            entropy_tensor = torch.tensor(entropy).reshape(1, 1)
            entropy_tensor = trans_to_cuda(entropy_tensor)
            similarity = trans_to_cuda(similarity)
            combined_tensor = torch.cat((similarity, entropy_tensor), dim=1)

            h.append(combined_tensor)
        h = torch.cat(h, dim=0)

        return h

    # 划分会话物品
    def focused_or_wandering_space_item(self, sessions, session_len, p_intent_tensor, opt):
        focused_items_list = []
        wandering_items_list = []
        binary_matrix = self.binary_matrix

        for i, session in enumerate(sessions):
            current_session_matrix = binary_matrix[session]
            # 计算每个物品所属主题的分布信息熵
            entropy_list = []
            for row in current_session_matrix:
                ones_count = np.sum(row)
                ones_prob = ones_count / len(row)
                if ones_prob == 0:
                    ones_prob = 1e-10
                entropy = -ones_prob * np.log2(ones_prob) - (1 - ones_prob) * np.log2(1 - ones_prob)
                entropy_list.append(entropy)

            # 找到当前会话内信息熵大于阈值的物品索引
            threshold = self.threshold.item()
            wandering_items_indicates = [i for i, entropy in enumerate(entropy_list) if entropy > threshold]

            current_session_vector = p_intent_tensor[i, :-1]
            similarities = current_session_vector[:session_len[i]]
            min_index = torch.argmin(similarities)
            wandering_items_indicates.append(min_index)

            wandering_items = [session[idx] for idx in wandering_items_indicates]
            wandering_items = list(set(wandering_items))

            wandering_items_list.append(wandering_items)

        for i, session in enumerate(sessions):
            focused_items = list(set(session) - set(wandering_items_list[i]))
            focused_items_list.append(focused_items)
        return focused_items_list, wandering_items_list

    # 集中购物
    def focused_space(self, sessions, session_len, max_n_node, opt):
        # 遍历当前批次的每个会话
        # for i, session in enumerate(sessions):
        #     # 当前会话实际的长度
        #     len = session_len[i]
        #     session = session[:len[0]]
        #     item_indices = [item.item() - 1 for item in session]
        #
        #     topic_lists = []
        #     # 遍历当前会话的每个物品，找到每个物品属于的主题列表
        #     for item in item_indices:
        #         # Find indices of topics for the current item
        #         topic_indices = np.where(self.binary_matrix[item] == 1)[0]
        #         topic_lists.append(topic_indices)
        #     topics = [topic for sublist in topic_lists for topic in sublist]
        #     topics = list(set(topics))  # 主题去重
        #
        #     # 对每个主题，找到属于这个主题的物品
        #     items_in_topics = [list(np.where(self.binary_matrix[:, topic] == 1)[0]) for topic in topics]
        #
        #     items_in_topics = np.asarray(items_in_topics)  # 转换为NumPy数组
        #     H_T = focused_data_masks(items_in_topics, self.n_node)
        #     BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        #     BH_T = BH_T.T  # (len(all_sessions),n_node)
        #
        #     H = H_T.T  # 稀疏矩阵，只有0和1(n_node，len(all_sessions))
        #     DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
        #     DH = DH.T  # (n_node, len(all_sessions))
        #     DHBH_T = np.dot(DH, BH_T)  # (n_node，n_node)
        #     adjacency = DHBH_T.tocoo()  # (n_node，n_node)
        #
        #     values = adjacency.data
        #     indices = np.vstack((adjacency.row, adjacency.col))
        #     i = torch.LongTensor(indices)
        #     v = torch.FloatTensor(values)
        #     shape = adjacency.shape
        #     adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        #     self.adjacency_fosued = adjacency  # 邻接矩阵 (n_node, n_node)
        #
        #     # 得到经过超图卷积后当前会话的物品表示
        #     item_embeddings_hg = self.HyperGraph(self.adjacency_fosued, self.embedding.weight)
        #     # print("得到经过超图卷积后当前会话的物品表示")

        # 全局级建模
        if opt.controller_way == 1:
            binary_matrix_T = self.binary_matrix.T  # (num_topics, n_node)
            items_in_topics = np.asarray(binary_matrix_T)  # 转换为NumPy数组
            H_T = focused_data_masks(items_in_topics, self.n_node)
            BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
            BH_T = BH_T.T  # (len(all_sessions),n_node)

            H = H_T.T  # 稀疏矩阵，只有0和1(n_node，len(all_sessions))
            DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
            DH = DH.T  # (n_node, len(all_sessions))
            DHBH_T = np.dot(DH, BH_T)  # (n_node，n_node)
            adjacency = DHBH_T.tocoo()  # (n_node，n_node)

            values = adjacency.data
            indices = np.vstack((adjacency.row, adjacency.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = adjacency.shape
            adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
            self.adjacency_fosued = adjacency  # 邻接矩阵 (n_node, n_node)

            # 得到经过超图卷积后当前会话的物品表示
            item_embeddings_hg = self.HyperGraph(self.adjacency_fosued, self.embedding.weight)

        elif opt.controller_way == 2:
            # # 遍历当前批次的每个会话
            # sessions1 = []
            # for i, session in enumerate(sessions):
            #     # 当前会话实际的长度
            #     len = session_len[i]
            #     session = session[:len[0]]
            #     item_indices = [item.item() - 1 for item in session]  # 从0开始
            #     sessions1.append(item_indices)

            # 遍历当前批次的每个会话
            for i, session in enumerate(sessions):
                # # 当前会话实际的长度
                len = session_len[i]
                session = session[:len[0]]
                item_indices = [item.item() - 1 for item in session]
                # 建立物品和它的id的映射字典
                item2index = {item: idx for idx, item in enumerate(item_indices)}

                binary_matrix = self.binary_matrix
                sub_matrix = binary_matrix[item_indices]  # (4, 333)

                # 计算邻接矩阵,得到经过超图卷积后当前会话的物品表示
                sub_matrix = sub_matrix.T  # (num_topics, num_items)
                # 创建一个新的矩阵，行数不变，列数为所有物品的数量
                expanded_matrix = np.zeros((sub_matrix.shape[0], self.n_node))  # (num_topics, 65064)

                # 将原始矩阵的列复制到新矩阵的对应位置
                for item, index in item2index.items():
                    # 将原始矩阵的列复制到新矩阵的对应位置
                    expanded_matrix[:, item] = sub_matrix[:, index]

                raw = np.asarray(expanded_matrix)  # 转换为NumPy数组
                H_T = focused_data_masks(raw, self.n_node)  # 生成一个稀疏矩阵  (num_topics, self.n_node+1)

                BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
                BH_T = BH_T.T  # (len(edges),n_node)
                H = H_T.T
                DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
                DH = DH.T  # (n_node, len(edges))
                DHBH_T = np.dot(DH, BH_T)  # (n_node, n_node)
                adjacency = DHBH_T.tocoo()  # (n_node, n_node)
                values = adjacency.data
                indices = np.vstack((adjacency.row, adjacency.col))
                i = torch.LongTensor(indices)
                v = torch.FloatTensor(values)
                shape = adjacency.shape
                adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))

                self.adjacency1 = adjacency  # 邻接矩阵 (n_node, n_node)

            # 得到经过超图卷积后当前会话的物品表示
            item_embeddings_hg = self.HyperGraph(self.adjacency1, self.embedding.weight)

        # print("得到经过超图卷积后当前会话的物品表示")
        return item_embeddings_hg

    # 发散购物
    def wandering_space(self, sessions, intent_prop, k, wandering_items_list, is_wandering_lg, opt):
        item_embedding = self.embedding.weight
        # 得到每个会话的平均表征
        session_embeddings = torch.stack([torch.mean(item_embedding[session], dim=0) for session in sessions])

        # ############# 1.检索相似的邻居会话 #############
        if opt.neighbor_way == 1:

            intent_prop_list = intent_prop.squeeze().cpu().tolist()
            wandering_sessions_indices = [i for i, prob in enumerate(intent_prop_list) if prob > 0.5]

            wandering_sessions_list = []
            if len(wandering_sessions_indices) > 0:
                wandering_sessions_list = [sessions[i] for i in wandering_sessions_indices]

            # 对当前批次的每个会话
            all_neighbor_sessions_batch = []
            for i, session in enumerate(sessions):

                wandering_sessions = wandering_sessions_list

                wandering_items_sessions = []
                # # 得到可能是发散购物的物品
                items = wandering_items_list[i]
                for item in items:

                    wandering_items_sessions = [session for session in sessions if item in session]

                all_neighbor_sessions = wandering_sessions + wandering_items_sessions

                if len(all_neighbor_sessions) > 0:
                    all_neighbor_sessions = [list(x) for x in set(tuple(x) for x in all_neighbor_sessions) if
                                             x]

                if len(all_neighbor_sessions) < k:
                    need_length = k - len(all_neighbor_sessions)
                    # 根据重叠度检索相似会话
                    sessions_filter = [session for session in sessions if session not in all_neighbor_sessions]
                    similarities = calculate_session_similarity(session, sessions_filter)
                    top_k_sessions_indices = sorted(range(len(similarities)), key=lambda i: similarities[i],
                                                    reverse=True)[
                                             :need_length]
                    for idx in top_k_sessions_indices:
                        all_neighbor_sessions.append(sessions_filter[idx])
                    all_neighbor_sessions = [list(x) for x in set(tuple(x) for x in all_neighbor_sessions) if
                                             x]

                # 重新筛选邻居会话：两者都满足排在最前面
                elif len(all_neighbor_sessions) > k:
                    all_neighbor_sessions0 = all_neighbor_sessions
                    overlapping_session = [sublist for sublist in wandering_sessions if
                                           sublist in wandering_items_sessions]
                    all_neighbor_sessions = overlapping_session
                    if len(all_neighbor_sessions) < k:
                        need_length = k - len(all_neighbor_sessions)
                        non_overlapping_session = [sublist for sublist in all_neighbor_sessions0 if
                                                   sublist not in overlapping_session]
                        all_neighbor_sessions.extend(non_overlapping_session[:need_length])

                all_neighbor_sessions_batch.append(all_neighbor_sessions)

        elif opt.neighbor_way == 2:
            # 对当前批次的每个会话
            all_neighbor_sessions_batch = []
            all_select_items = []
            for i, session in enumerate(sessions):
                items = wandering_items_list[i]
                select_items_emb = item_embedding[items]
                select_items_emb1 = torch.mean(select_items_emb, dim=0).unsqueeze(0)
                all_select_items.append(select_items_emb1)
            all_select_items = torch.cat(all_select_items, dim=0)

            similarities = torch.matmul(all_select_items, session_embeddings.T)

            all_neighbor_sessions = []
            # all_neighbor_sessions_emb = []
            for row in similarities:
                top_k_indices = torch.argsort(row, descending=True)[:k]

                neighbor_sessions = [sessions[i.item()] for i in top_k_indices.cpu()]
                all_neighbor_sessions.append(neighbor_sessions)
            all_neighbor_sessions_batch = all_neighbor_sessions

        # ############# 2.当前会话和邻居会话共同建立超图 ############
        batch_sessions_embeddings_hg = []
        batch_wondering_session_emb_lg = []
        for idx, session in enumerate(sessions):
            neighbor_session = all_neighbor_sessions_batch[idx]
            if session not in neighbor_session:
                neighbor_session.append(session)

            items = list(set([item for s in neighbor_session for item in s]))
            item2index = {item: index for index, item in enumerate(items)}  # 物品-索引映射字典
            index_session = []
            for i_session in neighbor_session:
                index_s = []
                for i in i_session:
                    index_s.append(item2index[i])
                index_session.append(index_s)

            current_index = [item2index[i] for i in session]
            nodes = len(items)  # 节点个数

            part_items_emb = self.embedding.weight[torch.tensor(items)]

            # 建立超图
            items_in_topics = np.asarray(index_session)
            H_T = focused_data_masks(items_in_topics, nodes)
            BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
            BH_T = BH_T.T
            H = H_T.T
            DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
            DH = DH.T
            DHBH_T = np.dot(DH, BH_T)
            adjacency = DHBH_T.tocoo()

            values = adjacency.data
            indices = np.vstack((adjacency.row, adjacency.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = adjacency.shape
            adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
            self.adjacency_wandering = adjacency

            part_item_embeddings_hg = self.HyperGraph1(self.adjacency_wandering,part_items_emb)

            if opt.wandering_con_way == 1:
                current_item_embeddings_hg = part_item_embeddings_hg[current_index]
                current_session_embeddings_hg = torch.mean(current_item_embeddings_hg, dim=0).unsqueeze(0)
            elif opt.wandering_con_way == 2:
                wander_items = wandering_items_list[idx]
                wander_index = [item2index[i] for i in wander_items]
                current_item_embeddings_hg = part_item_embeddings_hg[wander_index]
                current_session_embeddings_hg = torch.mean(current_item_embeddings_hg, dim=0).unsqueeze(0)
            elif opt.wandering_con_way == 3:
                neighbor_session_embeddings_hg = []
                for neighbor in neighbor_session:
                    # if neighbor != session:
                    neighbor_index = [item2index[i] for i in neighbor]
                    neighbor_item_embeddings_hg = part_item_embeddings_hg[neighbor_index]
                    neighbor_session_embedding_hg = torch.mean(neighbor_item_embeddings_hg, dim=0).unsqueeze(0)
                    neighbor_session_embeddings_hg.append(neighbor_session_embedding_hg)
                neighbor_session_embeddings_hg = torch.cat(neighbor_session_embeddings_hg, dim=0)
                current_session_embeddings_hg = torch.mean(neighbor_session_embeddings_hg, dim=0).unsqueeze(0)

            batch_sessions_embeddings_hg.append(current_session_embeddings_hg)

        batch_sessions_embeddings_hg = torch.cat(batch_sessions_embeddings_hg, dim=0)  # torch.Size([batch, emb])

        wandering_con_loss = 0.
        if is_wandering_lg is True:
            batch_wondering_session_emb_lg = torch.cat(batch_wondering_session_emb_lg, dim=0)
            wandering_con_loss = self.SSL(batch_sessions_embeddings_hg, batch_wondering_session_emb_lg)
        return batch_sessions_embeddings_hg, wandering_con_loss

    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)

        get = lambda i: item_embedding[reversed_sess_item[i]]

        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)

        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)

        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))

        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = score(sess_emb_hgnn, sess_emb_lgcn)
        neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
        one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)

        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss

    def forward(self, session_item, session_len, D, A, reversed_sess_item, mask, opt, max_n_node, p):
        sessions = []
        for i, session in enumerate(session_item):
            len = session_len[i]
            session = session[:len[0]]
            item_indices = [item.item() - 1 for item in session]
            sessions.append(item_indices)

        # 得到用户意图分布到两个意图空间的概率
        if opt.graph_way == 1:
            p_intent = self.intent_prob(session_item, session_len, max_n_node, opt)
        elif opt.graph_way == 2:
            p_intent = self.intent_prob1(sessions, session_len, max_n_node, opt, p)

        if opt.behavior_way == 1:
            p_intent_tensor = trans_to_cuda(torch.tensor(p_intent, dtype=torch.float32))
            self.h_size = max_n_node
            self.probability_network = trans_to_cuda(
                ProbabilityNetwork(self.h_size + 1, opt.prob_network_layer))  # 意图概率判别网络
            intent_prop = self.probability_network(p_intent_tensor)

        focused_items_list, wandering_items_list = self.focused_or_wandering_space_item(sessions, session_len,
                                                                                        p_intent_tensor, opt)
        # ######################################受控意图空间建模############################################
        item_embeddings_hg = self.embedding.weight
        item_embeddings_hg = self.focused_space(session_item, session_len, max_n_node, opt)
        focused_sess_emb = self.generate_sess_emb(item_embeddings_hg, session_item, session_len, reversed_sess_item,mask)

        # ########################################非受控意图空间建模############################################
        wandering_con_loss = 0.
        if opt.is_non_controller is True:
            wandering_sess_emb, wandering_con_loss = self.wandering_space(sessions, intent_prop, opt.neigh_k,
                                                                          wandering_items_list, opt.is_wandering_lg,
                                                                          opt)
        # ####################################融合两个意图空间建模结果############################################
        non_control_prop = intent_prop[0, 0].item()

        if opt.prop_way == 1:
            non_control_prop = non_control_prop
            control_prop = 1 - non_control_prop
        elif opt.prop_way == 2:
            control_prop = 0.5
            non_control_prop = 0.5

        if opt.recommend == "emb_cn":
            focused_sess_emb = focused_sess_emb.view(-1, self.emb_size)
            sess_emb_h = control_prop * focused_sess_emb + non_control_prop * wandering_sess_emb

        elif opt.recommend == "score_cn":
            sess_emb_h = (focused_sess_emb, wandering_sess_emb, control_prop, non_control_prop)

        if opt.is_batch_norm is True:
            item_embeddings_hg = self.bn1(item_embeddings_hg)
        if opt.is_dropout is True:
            item_embeddings_hg = self.dropout15(item_embeddings_hg)

        if opt.is_merge is True:
            sess_emb_h = self.merge(sess_emb_h)

        return item_embeddings_hg, sess_emb_h, self.beta * (wandering_con_loss)

    def bpr_loss(self, sess_emb_hgnn, tar, neg_samples, item_emb_hg):
        pos_scores = torch.sum(sess_emb_hgnn * item_emb_hg[tar], dim=1)  # Calculate scores for positive samples
        neg_scores = torch.sum(sess_emb_hgnn * item_emb_hg[neg_samples], dim=1)  # Calculate scores for negative samples
        epsilon = 1e-8  # Small epsilon value
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + epsilon))  # Calculate BPR loss
        # print("bpr_loss", bpr_loss)
        return bpr_loss


@jit(nopython=True)
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    # k_largest_scores = [item[0] for item in n_candidates]
    return ids  # , k_largest_scores


def forward(model, i, data, opt, p):
    tar, session_len, session_item, reversed_sess_item, mask, max_n_node = data.get_slice(i)

    A_hat, D_hat = data.get_overlap(session_item)

    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())

    item_emb_hg, sess_emb_hgnn, con_loss = model(session_item, session_len, D_hat, A_hat, reversed_sess_item, mask,
                                                 opt, max_n_node, p)
    if (opt.recommend == "score_cn") and (opt.is_controller is True) and (opt.is_non_controller is True):
        focused_sess_emb, wandering_sess_emb, control_prop, non_control_prop = sess_emb_hgnn

        focused_sess_emb = focused_sess_emb.view(-1, opt.embSize)
        focused_scores = torch.mm(focused_sess_emb, torch.transpose(item_emb_hg, 1, 0))

        wandering_sess_emb = wandering_sess_emb.view(-1, opt.embSize)
        wandering_scores = torch.mm(wandering_sess_emb, torch.transpose(item_emb_hg, 1, 0))

        scores = control_prop * focused_scores + non_control_prop * wandering_scores
    else:
        scores = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
    return tar, scores, con_loss


def train_test(model, train_data, test_data, K, opt, p):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0

    slices = train_data.generate_batch(model.batch_size)
    # 对每个批次
    for i in slices:
        model.zero_grad()
        targets, scores, con_loss = forward(model, i, train_data, opt, p)
        loss = model.loss_function(scores + 1e-8, targets)
        if opt.is_regulation is True:
            reg_loss = 0.
            for param in model.parameters():
                reg_loss += torch.norm(param, p=2)
            loss = loss + opt.lamb * reg_loss

        loss = loss + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)

    top_K = [5, 10, 15, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []

    print('start predicting: ', datetime.datetime.now())

    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar, scores, con_loss = forward(model, i, test_data, opt, p)
        scores = trans_to_cpu(scores).detach().numpy()
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()

        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss

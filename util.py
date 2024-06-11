import csv

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix

import datetime
import pickle

from tqdm.auto import tqdm


# 输出当前时间
def printbar(s):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "=====" * 5 + s + ":" + "%s" % nowtime)


# 处理原始会话数据集中的历史数据
def str2list(x):
    x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
    l = [i for i in x.split() if i]
    return l


def pick_dump(obj, dir):
    with open(dir, 'wb') as f:
        pickle.dump(obj, f)


def pick_load(dir):
    with open(dir, 'rb') as f:
        obj = pickle.load(f)
    return obj


# 转换为稀疏矩阵表示
def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i] - 1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))
    return matrix


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def calculate_session_similarity(current_session, all_sessions):
    similarities = []
    for session in all_sessions:
        set_a = set(current_session)
        set_b = set(session)
        overlap = len(set_a & set_b)
        ab_set = set_a | set_b
        similarity = float(overlap) / float(len(ab_set))
        similarities.append(similarity)
    return similarities


class Data:
    def __init__(self, data, probs, shuffle=False, n_node=None):
        self.data = data
        self.probs = probs
        self.raw = np.asarray(data[0])

        H_T = data_masks(self.raw, n_node)
        BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)

        self.adjacency = DHBH_T.tocoo()

        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]

        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)

        session_len = []  # 每个会话的实际长度
        reversed_sess_item = []
        mask = []
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1] * len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
        return self.targets[index] - 1, session_len, items, reversed_sess_item, mask, max_n_node


def yu_process(sessions_data):
    all_sessions_his_data = []
    all_sessions_label = []
    data = sessions_data
    for _, row in tqdm(data.iterrows(), total=len(data)):
        prev_items = str2list(row['prev_items'])
        next_item = row['next_item']
        all_sessions_his_data.append(prev_items)
        all_sessions_label.append(next_item)

    all_sessions_str = []
    for idx, (his, label) in enumerate(zip(all_sessions_his_data, all_sessions_label)):
        his.append(label)
        all_sessions_str.append(his)

    output_file = 'datasets/Amazon_KDD/interactions.csv'

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Item'])
        for index, session in enumerate(all_sessions_str):
            for item in session:
                writer.writerow([index, item])


class Process:
    def __init__(self, data, k, m, opt):
        self.data = data
        self.k = k
        self.m = m
        self.opt = opt
        self.save_path = 'datasets/' + self.opt.dataset + '/'

        self.item2index = {}
        self.item2count = {}
        self.index2item = {}

        self.num_histories = 0
        self.num_words = 0

        self.all_sessions_index = []  # 根据映射字典创建数字编号格式的交互数据

    def filter_data(self):
        inter_data = pd.read_csv(self.data, header=0, names=['user_token', 'item_token'])

        if self.opt.dataset == 'Amazon_KDD':
            inter_data = inter_data.head(2000000)

        # 过滤掉被用户交互次数少于k次的物品所在的交互记录
        k = self.k
        item_counts = inter_data['item_token'].value_counts()
        filtered_data = inter_data[inter_data['item_token'].isin(item_counts[item_counts > k].index)]

        # 过滤掉用户交互次数少于m次的交互记录
        m = self.m
        user_counts = filtered_data['user_token'].value_counts()
        filtered_data1 = filtered_data[filtered_data['user_token'].isin(user_counts[user_counts > m].index)]

        data1 = filtered_data1
        data1.to_csv(self.save_path + "filtered_data.csv", header=None, index=False, encoding="utf-8")

        inter_dict = {}
        for _, row in tqdm(data1.iterrows(), total=len(data1)):
            user = row['user_token']
            item = row['item_token']
            inter_dict.setdefault(user, [])
            inter_dict[user].append(item)

        self.all_sessions = [items for user, items in inter_dict.items()]

        self.item2index = {}
        self.item2count = {}
        self.index2item = {}
        self.num_histories = 0
        self.num_words = 0
        # 建立物品-index映射字典
        for session in self.all_sessions:
            for item_str in session:
                self.num_histories += 1
                if item_str not in self.item2index:
                    self.item2index[item_str] = self.num_words + 1
                    self.item2count[item_str] = 1
                    self.index2item[self.num_words + 1] = item_str
                    self.num_words += 1
                else:
                    self.item2count[item_str] += 1

        self.all_sessions_index = []
        for session in self.all_sessions:
            sublist = []
            for item_str in session:
                sublist.append(self.item2index[item_str])
            self.all_sessions_index.append(sublist)

    # 划分数据集
    def split_data(self):
        # 划分所有会话数据为8:2
        self.pre_data = self.all_sessions_index[:int(0.8 * len(self.all_sessions_index))]
        self.post_data = self.all_sessions_index[int(0.8 * len(self.all_sessions_index)):]

        # #########得到训练集##########
        self.train_data = []
        # 对于前80%的数据,从第一个元素开始，每次取多个元素并将它们存储为一个列表添加到训练集中
        for session in self.pre_data:
            for i in range(len(session)):
                # 从第一个元素开始，每次取多个元素并添加到result_list中
                self.train_data.append(session[:i + 3])
        #  后20%的数据先把每个会话最后一个物品删除，剩下的数据再从第一个元素开始，每次取多个元素并将它们存储为一个列表添加到训练集中
        selected_data = []
        for session in self.post_data:
            selected_data.append(session[:-1])
        for session in selected_data:
            for i in range(len(session)):
                # 从第一个元素开始，每次取多个元素并添加到result_list中
                self.train_data.append(session[:i + 3])

        # ###########得到测试集############
        self.test_data = self.post_data

    # 保存数据
    def save(self):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        all_train_seq = []

        for session in self.train_data:
            train_x.append(session[:-1])
            train_y.append(session[-1])
            all_train_seq.append(session[:])

        self.train_set = (train_x, train_y)

        for session in self.test_data:
            test_x.append(session[:-1])
            test_y.append(session[-1])
        self.test_set = (test_x, test_y)

        # 打开文件以写入pickle数据

        with open(self.save_path + 'train.pkl', 'wb') as file:
            pickle.dump(self.train_set, file)

        # 打开文件以写入pickle数据
        with open(self.save_path + 'test.pkl', 'wb') as file:
            pickle.dump(self.test_set, file)

        # 打开文件以写入pickle数据
        with open(self.save_path + 'all_train_seq.pkl', 'wb') as file:
            pickle.dump(all_train_seq, file)

    def get_data(self):
        return self.train_data, self.test_data



import argparse
from bertopic import BERTopic

from model import *


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 所有物品标题进行主题建模
def topic_model(product_file, p, opt):
    products_data = pd.read_csv(product_file, sep=',', header=0)
    all_items_id = list(p.index2item.values())

    selected_data = products_data[products_data['id'].isin(all_items_id)]
    all_titles = selected_data['title'].tolist()

    if opt.is_stable_topics == 'auto':
        model = BERTopic(language='english', calculate_probabilities=True, verbose=True, nr_topics="auto")
    elif opt.is_stable_topics == 'dynamic':
        model = BERTopic(language='english', calculate_probabilities=True, verbose=True)

    topics, probs = model.fit_transform(all_titles)
    # 概率矩阵转换为只包含0和1的矩阵
    # 找到每行中前20个最大值的索引
    top_indices = np.argsort(-probs, axis=1)[:, :opt.filter_num_topics]
    binary_matrix = np.zeros(probs.shape)
    rows = np.arange(probs.shape[0])[:, None]
    binary_matrix[rows, top_indices] = 1
    return model, topics, probs, binary_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Amazon_KDD', help='dataset name: Amazon_KDD/Amazon_Beauty')
parser.add_argument('--epoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
parser.add_argument('--embSize', type=int, default=128, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

parser.add_argument("--top_k", type=int, default=20, help="compute metrics@top_k")

parser.add_argument('--graph_way', type=int, default=1, help='1/2')
parser.add_argument('--prob_network_layer', type=int, default=2, help='the number of prob_network_layer')  # 1一层/2拼接后再经过一层Linear层
parser.add_argument('--item_threshold', type=int, default=1)  # 判断物品是发散的阈值

parser.add_argument('--controller_way', type=int, default=1, help='1/2')  # 1.全局/2.当前会话
parser.add_argument('--is_stable_topics', default='stable', help='stable/dynamic/auto')  # 主题数设置
parser.add_argument('--filter_num_topics', type=int, default=20)  # 过滤后的每个物品属于的主题数

parser.add_argument('--neighbor_way', type=int, default=2, help='1/2')
parser.add_argument("--neigh_k", type=int, default=10, help="neighbour number")
parser.add_argument('--wandering_con_way', type=int, default=1, help='1/2/3')
parser.add_argument('--is_wandering_lg', type=bool, default=True)

parser.add_argument('--recommend', default='emb_cn', help='emb_cn/score_cn')
parser.add_argument('--prop_way', type=int, default=1, help='1/2')
parser.add_argument('--is_linear', type=bool, default=False)
parser.add_argument('--is_merge', type=bool, default=False)
parser.add_argument('--is_batch_norm', type=bool, default=False)
parser.add_argument('--is_dropout', type=bool, default=False)


parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--is_regulation', type=bool, default=False, help='lamb regularization weight')
parser.add_argument('--lamb', type=float, default=1e-5, help='lamb regularization weight')
parser.add_argument('--loss', default="CrossEntropy", help='CrossEntropy/bpr')

opt = parser.parse_args()
print(opt)


def main():
    init_seed(2023)

    sessions_data = pd.read_csv(r'datasets/Amazon_KDD/sessions_UK.csv', header=0)
    yu_process(sessions_data)

    file_path = 'datasets/' + opt.dataset + '/'
    output_file = file_path + 'interactions.csv'
    product_file = file_path + 'products_UK.csv'
    if opt.is_stable_topics == 'auto':
        bertopic_file = file_path + 'topic_probs.pkl'
    elif opt.is_stable_topics == 'stable':
        bertopic_file = file_path + 'topic_probs' + str(opt.num_topics) + '.pkl'

    if opt.dataset == 'Amazon_KDD':
        p = Process(output_file, 10, 10, opt)
    elif opt.dataset == 'Amazon_Beauty':
        p = Process(output_file, 1, 1, opt)

    p.filter_data()
    p.split_data()
    p.save()

    bertopic_model, topics, probs, binary_matrix = topic_model(product_file, p, opt)

    train_data = pick_load(file_path + 'train.pkl')

    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        train_data = train_data
        test_data = pick_load(file_path + 'test.pkl')

    if opt.dataset == 'Amazon_KDD':
        n_node = 21839
    elif opt.dataset == 'Amazon_Beauty':
        n_node = 4379

    train_data = Data(train_data, probs, shuffle=True, n_node=n_node)
    test_data = Data(test_data, probs, shuffle=True, n_node=n_node)
    model = trans_to_cuda(
        DbMei(adjacency=train_data.adjacency, n_node=n_node, lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=opt.layer,
              emb_size=opt.embSize, batch_size=opt.batchSize, dataset=opt.dataset,
              probs=probs, binary_matrix=binary_matrix))

    top_K = [5, 10, 15, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch + 1)
        metrics, total_loss = train_test(model, train_data, test_data, opt.top_k, opt, p)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch + 1
                # 保存模型
                torch.save(model.state_dict(), opt.dataset + 'DHCN.pth')

            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch + 1
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))

        torch.cuda.empty_cache()  # 清理GPU缓存


def pop():
    file_path = 'datasets/' + opt.dataset + '/'
    output_file = file_path + 'interactions.csv'

    if opt.dataset == 'Amazon_KDD':
        p = Process(output_file, 10, 10, opt)
    elif opt.dataset == 'Amazon_Beauty':
        p = Process(output_file, 1, 1, opt)

    p.filter_data()
    p.split_data()
    p.save()

    if opt.dataset == 'Amazon_KDD':
        n_node = 62593
    elif opt.dataset == 'Amazon_Beauty':
        n_node = 4379

    train_data = pick_load(file_path + 'train.pkl')

    if opt.validation:

        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        train_data = train_data
        test_data = pick_load(file_path + 'test.pkl')

    test_x, test_y = test_data

    # 得到每个物品的交互次数
    item_counts = p.computer_item_counts()
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
    # 选择前N个物品作为热门物品列表
    popular_items = [item[0] for item in sorted_items[:n_node]]

    top_K = [5, 10, 15, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []

    for k in top_K:
        for session, target in zip(test_x, test_y):
            recommendations = popular_items[:k]
            rank = np.where(recommendations == target)[0]
            if len(rank) > 0:
                metrics['hit%d' % k].append(np.isin(target, recommendations))
            else:
                metrics['hit%d' % k].append(np.isin(target, recommendations))

            rank = 0
            for i, rec in enumerate(recommendations):
                if rec == target:
                    rank = 1 / (i + 1)
                    break
            metrics['mrr%d' % k].append(rank)

    for K in top_K:
        metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
        metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
        print('Recall@%d: %.4f\tMRR@%d: %.4f\t' % (K, metrics['hit%d' % K], K, metrics['mrr%d' % K]))


if __name__ == '__main__':
    main()
    # pop()

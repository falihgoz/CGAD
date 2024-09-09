import src.te.te_compute as te
import numpy as np
import random
import time
import torch
import os

def normalize_edge(edge_weight):
    if edge_weight.shape[0] > 1:
        edge_min, edge_max = edge_weight.min(), edge_weight.max()

        if edge_min != edge_max:
            new_min, new_max = 0, 1
            normal_edge_weight = (edge_weight - edge_min) / ((edge_max - edge_min) + 0.00001)
            normal_edge_weight = (normal_edge_weight * (new_max - new_min)) + new_min
        else:
            normal_edge_weight = edge_weight / edge_max
    else:
        normal_edge_weight = edge_weight

    return normal_edge_weight

def build_causal_graph(dataset, subset):
    data_dir = f'./preprocessed_data/{dataset}'
    train_data = np.load(f'{data_dir}/{subset}train.npy', allow_pickle=True)
    _max_processing_length = 2000
    _max_sampling = 15
    _min_weak_relation = 0.1
    _te_knn = 3

    len_features, num_nodes = train_data.shape

    processed_len = min(len_features, _max_processing_length)
    max_random = max(0, len_features - processed_len - 1)
    sampling_num = min(len_features // processed_len, _max_sampling)

    print('num nodes:', num_nodes)
    print('num features:', len_features)
    print('num sampling:', sampling_num)

    start_time = time.time()
    adj_matrix = np.empty(shape=(0, num_nodes, num_nodes))

    for k in range(sampling_num):
        adj_matrix_sample = []
        random_point = random.randint(0, max_random)

        for i in range(num_nodes):
            backward_relations = []

            for j in range(num_nodes):
                if i != j:
                    TE = te.te_compute(
                        train_data[random_point:random_point + processed_len, i],
                        train_data[random_point:random_point + processed_len, j],
                        k=_te_knn, embedding=1, safetyCheck=False, GPU=False
                    )
                    backward_relations.append(TE)
                else:
                    backward_relations.append(0)

            progress = 100 * ((i + 1) / num_nodes)

            if i % 20 == 0:
                print(f'Finished processing of {int(progress)}% in sample {k + 1}. Current progress is in node {i}')
            adj_matrix_sample.append(backward_relations)

        adj_matrix = np.append(adj_matrix, [adj_matrix_sample], axis=0)

    adj_matrix = np.mean(adj_matrix, axis=0)
    adj_torch = torch.from_numpy(adj_matrix)
    adj_torch = torch.where(adj_torch > _min_weak_relation, adj_torch, 0)

    edge_index = (adj_torch > 0).nonzero().t().contiguous()
    row, col = edge_index
    edge_weight = adj_torch[row, col]

    normal_edge_weight = normalize_edge(edge_weight)

    os.makedirs(data_dir, exist_ok=True)
    torch.save(edge_index, f'{data_dir}/{subset}edge_index.pt')
    torch.save(normal_edge_weight, f'{data_dir}/{subset}edge_weight.pt')

    print(f"{time.time() - start_time} seconds of processing time")
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import random
import os

from src.process_adj import build_causal_graph
from scipy.stats import rankdata, iqr
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data, sequence_length, batch_size, mode="Training"):
        """
        Initialize a TimeDataset.

        Args:
            raw_data (torch.Tensor): The raw data.
            sequence_length (int): Length of the input sequence.
            batch_size (int): The batch size.
            mode (str, optional): The mode (e.g., 'Training'). Default is 'Training'.
        """
        self.raw_data = raw_data
        self.sequence_length = sequence_length
        self.mode = mode

        # Convert raw_data to tensor with double precision
        data = torch.tensor(raw_data).double()

        self.x, self.y = self.process(data, batch_size)

    def __len__(self):
        return len(self.x)

    def process(self, data, batch_size):
        x_arr, y_arr = [], []

        node_num, total_time_len = data.shape
        print("-" * 89)
        print("mode:", self.mode)
        print("slide_win:", self.sequence_length)
        print("total_time_len:", total_time_len)

        for i in range(self.sequence_length, total_time_len):
            input_sequence = data[:, i - self.sequence_length : i]
            target = data[:, i]

            x_arr.append(input_sequence)
            y_arr.append(target)

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        if x.shape[0] % batch_size == 1:
            x = x[1:]
            y = y[1:]

        print("x:", x.shape)
        print("y:", y.shape)

        return x, y

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        target = self.y[idx].double()
        return feature, target


def prepare_data(dataset, subset, batch_size, seq_in_len):
    edge_index_loc = f"./preprocessed_data/{dataset}/{subset}edge_index.pt"
    edge_weight_loc = f"./preprocessed_data/{dataset}/{subset}edge_weight.pt"

    if not os.path.exists(edge_index_loc) and not os.path.exists(edge_weight_loc):
        build_causal_graph(dataset, subset)

    train = np.load(
        f"./preprocessed_data/{dataset}/{subset}train.npy", allow_pickle=True
    )
    test = np.load(f"./preprocessed_data/{dataset}/{subset}test.npy", allow_pickle=True)
    labels = np.load(
        f"./preprocessed_data/{dataset}/{subset}labels.npy", allow_pickle=True
    )

    train = np.transpose(train)
    test = np.transpose(test)

    edge_index = torch.load(edge_index_loc)
    edge_weight = torch.load(edge_weight_loc)
    num_nodes = len(train)

    train_dataset = TimeDataset(
        train, seq_in_len, batch_size=batch_size, mode="Training"
    )
    test_dataset = TimeDataset(test, seq_in_len, batch_size=batch_size, mode="Testing")

    val_start_index = int(0.80 * len(train_dataset))  # all is 80, try 0.95 for MSL

    train_indices = list(range(val_start_index))
    val_indices = list(range(val_start_index, len(train_dataset)))

    train_dataset_final = Subset(train_dataset, train_indices)
    val_dataset = Subset(train_dataset, val_indices)

    print("len val", len(val_dataset))
    labels = labels[-len(test_dataset.x) :]

    train_dataloader = DataLoader(
        train_dataset_final, batch_size=batch_size, shuffle=False
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        edge_index,
        edge_weight,
        num_nodes,
        labels,
    )


def result_to_excel(
    dataset,
    subset,
    precision,
    recall,
    auc,
    f1,
    tp,
    tn,
    fp,
    fn,
    lm,
    final_labels,
    pred,
    score,
):
    # Define the root directory for experiment results
    result_dir = "./experiment_results"

    # Store results in a CSV file
    result_data = {
        "subset": [subset],
        "P": [precision],
        "R": [recall],
        "AUC": [auc],
        "F1": [f1],
        "TP": [tp],
        "TN": [tn],
        "FP": [fp],
        "FN": [fn],
        "LM": [lm],
    }

    result_file_path = os.path.join(result_dir, f"{dataset}.csv")

    if not os.path.exists(result_file_path):
        result_df = pd.DataFrame(result_data)
        result_df.to_csv(result_file_path, index=False)
    else:
        new_row = pd.DataFrame(result_data)
        result_df = pd.read_csv(result_file_path)
        result_df = pd.concat([result_df, new_row], axis=0)
        result_df.to_csv(result_file_path, index=False)

    raw_result_file_path = os.path.join(result_dir, "raw", f"{dataset}_{subset}.csv")

    raw_result_data = {
        "Actual": final_labels,
        "Prediction": pred,
        "Anomaly Score": score,
    }
    raw_result_df = pd.DataFrame(raw_result_data)
    raw_result_df.to_csv(raw_result_file_path, index=False)

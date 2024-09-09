import time
import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from src.mtgnn import gtnet
from src.utils import *
from src.trainer import *
from src.detect_anomaly import *
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Using a target size .* that is different to the input size .*",
)


def create_edge_index(num_nodes):
    num_edges = int(
        num_nodes * (num_nodes - 1) / 2
    )  # Number of edges in a fully connected graph
    edge_index = np.zeros((2, num_edges), dtype=np.int64)

    idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_index[0, idx] = i
            edge_index[1, idx] = j
            idx += 1

    return edge_index


def train_and_evaluate(
    model,
    device,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    epochs,
    save,
):
    best_val = float("inf")

    try:
        print("begin training")
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            train_loss = train(model, device, train_dataloader)
            val_loss = evaluate(model, device, val_dataloader)

            print(
                "| end of epoch {:3d} | time: {:5.2f}s | train_mse_loss {:5.4f} | valid_mse_loss {:5.4f}".format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss
                ),
                flush=True,
            )

            if val_loss < best_val and epoch > 1:
                with open(save, "wb") as f:
                    torch.save(model, f)
                best_val = val_loss

            if epoch % 5 == 0:
                test_mse_loss = evaluate(model, device, test_dataloader)
                print(
                    "test_mse_loss {:5.4f}".format(test_mse_loss),
                    flush=True,
                )

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")


def start(args):
    dataset = args.dataset
    subset = args.subset or {
        "SMD": "machine-1-1",
        "SMAP": "P-1",
        "MSL": "C-1",
    }.get(dataset, "")

    print("-" * 90)
    print("Dataset", dataset)
    print("Subset", subset)

    save = (
        f"./checkpoints/{dataset}_{subset}.pt"
        if subset
        else f"./checkpoints/{dataset}.pt"
    )

    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        edge_index,
        edge_weight,
        num_nodes,
        labels,
    ) = prepare_data(
        dataset, f"{subset}_" if subset else "", args.batch_size, args.seq_in_len
    )

    print("edge_index:", edge_index.shape)
    print("edge_weight:", edge_weight.shape)
    print("labels:", labels.shape)

    gcn_true = False if edge_weight.shape[0] == 0 else True

    model = gtnet(
        args.device, num_nodes, edge_index, edge_weight, args.seq_in_len, gcn_true
    )
    model = model.to(args.device)

    nParams = sum([p.nelement() for p in model.parameters()])

    print("-" * 89)
    print("Number of model parameters is", nParams, flush=True)

    if not os.path.exists(save) or args.retrain:
        train_and_evaluate(
            model,
            args.device,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            args.epochs,
            save,
        )

    # Load the best saved model.
    with open(save, "rb") as f:
        model = torch.load(f)

    # Test
    best_test_loss, test_result = test(model, args.device, test_dataloader)
    final_val_mse, val_result = test(model, args.device, val_dataloader)

    print("-" * 89)
    print(
        "test_mse_loss {:5.4f} | val_mse_loss {:5.4f}".format(
            best_test_loss, final_val_mse
        )
    )

    detect_anomaly(test_result, val_result, labels, args.seq_in_len, args.metrics)


def create_directories(folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


if __name__ == "__main__":
    folders = ["checkpoints", "experiment_results", "data", "preprocessed_data"]
    create_directories(folders)

    parser = argparse.ArgumentParser(
        description="Entropy Causal Graph for Multivariate Anomaly Detection"
    )
    parser.add_argument("--dataset", type=str, default="MSL", help="dataset location")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device (e.g., cuda:0)"
    )
    parser.add_argument(
        "--seq_in_len", type=int, default=15, help="input sequence length"
    )
    parser.add_argument("--subset", type=str, default="", help="sub-dataset")
    parser.add_argument("--seed", type=int, default=0, help="random stabilizer")
    parser.add_argument("--retrain", action="store_true", help="retraining")
    parser.add_argument(
        "--metrics",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Choose the metrics: 0 for point-adjust F1, 1 for point-wise F1, 2 for composite F1",
    )

    args = parser.parse_args()

    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    start(args)

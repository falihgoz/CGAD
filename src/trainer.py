import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def test(model, device, test_dataloader):
    loss_func = nn.MSELoss(reduction="sum").to(device)
    total_loss_mse = 0
    t_test_predicted_list, t_test_ground_list = [], []

    for X, Y in test_dataloader:
        X, Y = [item.to(device, dtype=torch.float) for item in [X, Y]]

        with torch.no_grad():
            X = torch.unsqueeze(X, dim=1)
            output = model(X)
            output = torch.squeeze(output)

            loss = loss_func(output, Y)
            total_loss_mse += loss.item()

            if output.ndimension() == 1:
                output = output.view(1, -1)

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = output
                t_test_ground_list = Y
            else:
                t_test_predicted_list = torch.cat(
                    (t_test_predicted_list, output), dim=0
                )
                t_test_ground_list = torch.cat((t_test_ground_list, Y), dim=0)

    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()

    avg_loss_mse = total_loss_mse / len(test_dataloader)
    return avg_loss_mse, [test_predicted_list, test_ground_list]


def evaluate(model, device, eval_dataloader):
    loss_func = nn.MSELoss(reduction="sum").to(device)
    model.eval()
    total_loss_mse = 0

    for X, Y in eval_dataloader:
        X, Y = [item.to(device, dtype=torch.float) for item in [X, Y]]

        with torch.no_grad():
            X = torch.unsqueeze(X, dim=1)
            output = model(X)
            output = torch.squeeze(output)
            loss = loss_func(output, Y)
            total_loss_mse += loss.item()

    avg_loss_mse = total_loss_mse / len(eval_dataloader)
    return avg_loss_mse


def train(model, device, train_dataloader):
    loss_func = nn.MSELoss(reduction="sum").to(device)
    model.train()
    total_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    for X, Y in train_dataloader:
        X, Y = [item.to(device, dtype=torch.float) for item in [X, Y]]
        optimizer.zero_grad()

        X = torch.unsqueeze(X, dim=1)
        output = model(X)
        output = torch.squeeze(output)

        loss = loss_func(output, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

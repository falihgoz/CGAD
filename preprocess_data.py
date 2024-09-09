import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from shutil import copyfile
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
import csv
import ast
from pickle import dump

datasets = ["SMD", "SWAT", "SMAP", "MSL", "WADI"]
output_folder = "preprocessed_data"
data_folder = "data"


def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(
        os.path.join(dataset_folder, category, filename),
        dtype=np.float64,
        delimiter=",",
    )
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape


def load_and_save2(category, filename, dataset, dataset_folder, shape):
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, "interpretation_label", filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(":")[0], line.split(":")[1].split(",")
        start, end, indx = (
            int(pos.split("-")[0]),
            int(pos.split("-")[1]),
            [int(i) - 1 for i in values],
        )
        temp[start - 1 : end - 1, indx] = 1
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)


def norm(train, test):
    normalizer = MinMaxScaler(feature_range=(0, 1)).fit(train)
    train_ret = normalizer.transform(train)
    test_ret = normalizer.transform(test)
    return train_ret, test_ret


def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return a / 2 + 0.5


def normalize2(a, min_a=None, max_a=None):
    if min_a is None:
        min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a), min_a, max_a


def normalize3(a, min_a=None, max_a=None):
    if min_a is None:
        min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return (a - min_a) / (max_a - min_a + 0.0000001), min_a, max_a


def convertNumpy(df):
    x = df[df.columns[3:]].values[:, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)


def transformstr(df):
    for i in list(df):
        df[i] = df[i].apply(lambda x: str(x).replace(",", "."))
    df = df.astype(float)
    return df


def load_data(dataset):
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    if dataset == "SMD":
        dataset_folder = "data/SMD"
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith(".txt"):
                load_and_save("train", filename, filename.strip(".txt"), dataset_folder)
                s = load_and_save(
                    "test", filename, filename.strip(".txt"), dataset_folder
                )
                load_and_save2(
                    "labels", filename, filename.strip(".txt"), dataset_folder, s
                )
    elif dataset == "SWAT":
        dataset_folder = "data/SWAT/"
        normal = pd.read_parquet(
            os.path.join(dataset_folder, "SWaT_Dataset_Normal_v1.parquet")
        )
        attack = pd.read_parquet(
            os.path.join(dataset_folder, "SWaT_Dataset_Attack_v0.parquet")
        )
        labels = (attack["Normal/Attack"] == "Attack").astype(int)
        y_true = labels.to_numpy() == 1

        attacks_ts = pd.read_csv(
            os.path.join(dataset_folder, "SWaT_Dataset_v0_attacks_timestamps.csv"),
            parse_dates=["StartTime", "EndTime"],
            date_format=" %d/%m/%Y %H:%M:%S",
        )

        attacks_ts["StartTime"] = pd.to_datetime(attacks_ts["StartTime"])
        attacks_ts["EndTime"] = pd.to_datetime(attacks_ts["EndTime"])

        y_true_ts = np.zeros(len(labels))
        gt_intervals = []
        index = list(attack.index)
        for _, (onset, offset) in attacks_ts.iterrows():
            onset = index.index(onset)
            offset = index.index(offset) + 1
            y_true_ts[onset:offset] = 1
            gt_intervals.append((onset, offset))
        y_true_ts.mean()
        y_true = y_true_ts == 1

        print("Contamination rate:", f"{y_true.mean()*100:.2f}")
        print("Number of anomalous events:", len(gt_intervals))
        event_lengths = np.diff(gt_intervals).reshape(-1)
        print("Min event length:", np.min(event_lengths))
        print("Max event length:", np.max(event_lengths))
        print("Average event length:", round(np.mean(event_lengths)))
        print("Median event length:", round(np.median(event_lengths)))

        y_true = y_true.astype(int)

        normal = normal.drop("Normal/Attack", axis=1)
        attack = attack.drop("Normal/Attack", axis=1)

        train = np.array(normal)
        test = np.array(attack)
        labels = np.broadcast_to(np.array(y_true).reshape(-1, 1), test.shape)

        print(train.shape, test.shape, labels.shape)
        for file in ["train", "test", "labels"]:
            np.save(os.path.join(folder, f"{file}.npy"), eval(file))

    elif dataset in ["SMAP", "MSL"]:
        dataset_folder = "data/SMAP_MSL"
        file = os.path.join(dataset_folder, "labeled_anomalies.csv")
        values = pd.read_csv(file)
        values = values[values["spacecraft"] == dataset]
        filenames = values["chan_id"].values.tolist()
        for fn in filenames:
            train = np.load(f"{dataset_folder}/train/{fn}.npy")
            test = np.load(f"{dataset_folder}/test/{fn}.npy")
            last_index_train = len(train) - 1
            merged_train_test = np.concatenate((train, test), axis=0)
            print(merged_train_test.shape)
            for i in range(len(merged_train_test[0])):
                merged_train_test[:, i] = minmax_scale(merged_train_test[:, i])

            train, test = (
                merged_train_test[: last_index_train + 1, :],
                merged_train_test[last_index_train + 1 :, :],
            )

            np.save(f"{folder}/{fn}_train.npy", train)
            np.save(f"{folder}/{fn}_test.npy", test)
            labels = np.zeros(test.shape)
            indices = values[values["chan_id"] == fn]["anomaly_sequences"].values[0]
            indices = indices.replace("]", "").replace("[", "").split(", ")
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels[indices[i] : indices[i + 1], :] = 1
            np.save(f"{folder}/{fn}_labels.npy", labels)

    elif dataset == "WADI":
        dataset_folder = "data/WADI"
        ls = pd.read_csv(os.path.join(dataset_folder, "WADI_attacklabels.csv"))
        train = pd.read_csv(os.path.join(dataset_folder, "WADI_14days.csv"))
        test = pd.read_csv(os.path.join(dataset_folder, "WADI_attackdata.csv"))

        print(train.shape)
        train.dropna(how="all", inplace=True)
        test.dropna(how="all", inplace=True)
        train.fillna(0, inplace=True)
        test.fillna(0, inplace=True)

        test["Time"] = test["Time"].astype(str)
        date_format = "%m/%d/%Y"
        time_format = "%I:%M:%S.%f %p"

        test["Time"] = pd.to_datetime(
            test["Date"] + " " + test["Time"], format=date_format + " " + time_format
        )

        labels = test.copy(deep=True)
        for i in test.columns.tolist()[3:]:
            labels[i] = 0
        for i in ["Start Time", "End Time"]:
            ls[i] = ls[i].astype(str)
            ls[i] = pd.to_datetime(ls["Date"] + " " + ls[i])
        for index, row in ls.iterrows():
            to_match = row["Affected"].split(", ")
            matched = []
            for i in test.columns.tolist()[3:]:
                for tm in to_match:
                    if tm in i:
                        matched.append(i)
                        break

            st, et = str(row["Start Time"]), str(row["End Time"])
            labels.loc[(labels["Time"] >= st) & (labels["Time"] <= et), matched] = 1

        train, test, labels = (
            convertNumpy(train),
            convertNumpy(test),
            convertNumpy(labels),
        )

        print(train.shape, test.shape, labels.shape)
        for file in ["train", "test", "labels"]:
            np.save(os.path.join(folder, f"{file}.npy"), eval(file))


if __name__ == "__main__":
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            load_data(d)
    else:
        print("Usage: python preprocess.py <datasets>")
        print(f"where <datasets> is space separated list of {datasets}")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.preprocessing import minmax_scale
from src.spot import SPOT
from scipy.stats import skew


def min_max(scores, min=None, max=None):
    epsilon = 1e-8
    if min == None or max == None:
        min = np.min(scores)
        max = np.max(scores)
        scores = (scores - min) / (max - min + epsilon)
    else:
        scores = (scores - min) / (max - min + epsilon)
    return scores, min, max


def mad(score):
    epsilon = 1e-5
    median = np.median(score)
    centered_data = score - median
    mad = np.median(np.abs(centered_data))
    score = centered_data / (mad + epsilon)
    return score


def topk(row, pr=False):
    k = int(np.sqrt(row.shape))
    kth_largest = np.partition(row, -k)[-k]
    mask = row >= kth_largest
    sum_of_top_k = row[mask].sum()
    return sum_of_top_k


def detect_anomaly(test_result, val_result, labels, window, metrics):
    num_nodes = len(test_result[0][0])

    test_result = np.array(test_result)
    val_result = np.array(val_result)

    test_error_all = None
    val_error_all = None

    for i in range(num_nodes):
        test_prediction_i = np.array(test_result[:2, :, i])
        val_prediction_i = np.array(val_result[:2, :, i])

        test_error_i = get_error(test_prediction_i)
        val_error_i = get_error(val_prediction_i)

        if test_error_all is None:
            test_error_all = test_error_i
        else:
            test_error_all = np.vstack((test_error_all, test_error_i))

        if val_error_all is None:
            val_error_all = val_error_i
        else:
            val_error_all = np.vstack((val_error_all, val_error_i))

    test_anomaly_score = np.max(test_error_all, axis=0)
    val_anomaly_score = np.max(val_error_all, axis=0)

    test_anomaly_score, _, _ = min_max(test_anomaly_score)
    val_anomaly_score, _, _ = min_max(val_anomaly_score)

    if labels.ndim > 1:
        labels = (np.sum(labels, axis=1) > 0.1) + 0

    pot_th = pot_eval(val_anomaly_score, test_anomaly_score)
    options = ["adjust_predicts", "point_wise", "composite"]

    ad_evaluation = options[metrics]

    if ad_evaluation == "adjust_predicts":
        best_th = get_best_th(
            test_anomaly_score, labels, pot_th, adjust_predicts, true_event=None
        )
        pred = adjust_predicts(test_anomaly_score, labels, threshold=best_th)
        f1, precision, recall, TP, TN, FP, FN, roc_auc = calc_point2point(pred, labels)
    elif ad_evaluation == "point_wise":
        best_th, pred = threshold_and_predict(test_anomaly_score, labels, ad_evaluation)
        f1, precision, recall, TP, TN, FP, FN, roc_auc = calc_point2point(pred, labels)
    else:
        true_events = get_events(labels)
        best_th, pred = threshold_and_predict(
            test_anomaly_score, labels, ad_evaluation, true_events
        )
        f1, precision, recall, TP, TN, FP, FN, roc_auc = calc_point2point(
            pred, labels, true_events
        )

    lm = best_th / pot_th

    print("-" * 89)
    print(f"F1 score: {f1}")
    print(f"AUC: {roc_auc}")
    print(f"precision: {precision}")
    print(f"recall: {recall}\n")
    print(f"Threshold: {pot_th}\n")
    print(f"Best Treshold: {best_th}")

    # return (
    #     f1,
    #     precision,
    #     recall,
    #     TP,
    #     TN,
    #     FP,
    #     FN,
    #     roc_auc,
    #     lm,
    #     labels,
    #     pred,
    #     test_anomaly_score,
    # )


def get_f_score(prec, rec):
    epsilon = 1e-8
    if prec == 0 and rec == 0:
        f_score = 0
    else:
        f_score = 2 * (prec * rec) / (prec + rec + epsilon)
    return f_score


def get_composite_fscore_from_scores(
    score_t_test, thres, true_events, prec_t, return_prec_rec=False
):
    epsilon = 1e-8
    pred_labels = score_t_test > thres
    tp = np.sum(
        [pred_labels[start : end + 1].any() for start, end in true_events.values()]
    )
    fn = len(true_events) - tp
    rec_e = tp / (tp + fn)
    fscore_c = 2 * (rec_e * prec_t) / (rec_e + prec_t + epsilon)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c


def threshold_and_predict(test_anomaly_score, labels, ad_evaluation, true_events=None):
    score_t_test = test_anomaly_score
    y_test = labels

    if ad_evaluation == "point_wise":
        prec, rec, thres = precision_recall_curve(y_test, score_t_test, pos_label=1)
        fscore = [
            get_f_score(precision, recall) for precision, recall in zip(prec, rec)
        ]
        opt_num = np.squeeze(np.argmax(fscore))
        opt_thres = thres[opt_num]
        pred_labels = np.where(score_t_test > opt_thres, 1, 0)

    else:
        prec, rec, thresholds = precision_recall_curve(
            y_test, score_t_test, pos_label=1
        )
        precs_t = prec
        fscores_c = [
            get_composite_fscore_from_scores(score_t_test, thres, true_events, prec_t)
            for thres, prec_t in zip(thresholds, precs_t)
        ]
        try:
            opt_thres = thresholds[np.nanargmax(fscores_c)]
        except:
            opt_thres = 0.0

    pred_labels = score_t_test > opt_thres
    return opt_thres, pred_labels


def get_best_th(anomaly_score, labels, th, operation, true_event=None):
    current_f1 = -1
    final_th = -1
    percent = 0
    step_num = 10000

    for i in range(step_num):
        th = np.percentile(anomaly_score, float(percent))
        pred = operation(anomaly_score, labels, threshold=th)
        f1, _, _, _, _, _, _, _ = calc_point2point(pred, labels, true_event)
        if f1 >= current_f1:
            current_f1 = f1
            final_th = th
        percent += 0.01

    return final_th


def get_error(error_result):
    pred, truth = error_result
    abs_error = np.square(
        np.subtract(
            np.array(pred).astype(np.float64), np.array(truth).astype(np.float64)
        )
    )
    abs_error = minmax_scale(abs_error)
    error = mad(abs_error)

    return error


def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def point_wise(score, label, threshold):
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    predict = score > threshold

    return predict


def calc_point2point(predict, actual, true_events=None):
    epsilon = 1e-8

    if true_events is not None:
        tp = np.sum(
            [predict[start : end + 1].any() for start, end in true_events.values()]
        )
        fn = len(true_events) - tp
    else:
        tp = np.sum(predict * actual)
        fn = np.sum((1 - predict) * actual)

    tn = np.sum((1 - predict) * (1 - actual))
    fp = np.sum(predict * (1 - actual))

    recall = tp / (tp + fn)
    precision = precision_score(actual, predict, zero_division=0)
    f1 = 2 * (recall * precision) / (recall + precision + epsilon)

    if precision == 0 and recall == 0:
        f1 = 0

    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0

    return f1, precision, recall, tp, tn, fp, fn, roc_auc


def pot_eval(init_score, score, q=1e-3, level=0.98):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
    Returns:
        dict: pot result dict
    """
    failing_count = 0

    while True:
        try:
            s = SPOT(q)  # SPOT object
            s.fit(init_score, score)  # data import
            s.initialize(
                level=level, min_extrema=False, verbose=False
            )  # initialization step
            ret = s.run(dynamic=False)  # run
            pot_th = np.mean(ret["thresholds"])
        except:
            level = level * 0.98
            failing_count += 1
            if failing_count > 10:
                pot_th = np.max(init_score) * 0.98
                break
        else:
            break

    return pot_th


def get_events(y_test, outlier=1, normal=0):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
        else:
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events


def get_composite_fscore_raw(pred_labels, true_events, y_test, return_prec_rec=False):
    epsilon = 1e-8
    tp = np.sum(
        [pred_labels[start : end + 1].any() for start, end in true_events.values()]
    )
    fn = len(true_events) - tp
    rec_e = tp / (tp + fn)
    prec_t = precision_score(y_test, pred_labels)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t + epsilon)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c

import json
import numpy as np
import tensorflow as tf
import pandas as pd


def precision_AT_K(actual, predicted, k, num_anomaly):
    act_set = np.array(actual[:k])
    pred_set = np.array(predicted[:k])
    ll = act_set & pred_set
    tt = np.where(ll == 1)[0]
    prec = len(tt) / float(k)
    rec = len(tt) / float(num_anomaly)
    return round(prec, 4), round(rec, 4)


def save_results(results, export_json):
    """Save results dict to a JSON-file."""
    with open(export_json, 'w') as fp:
        json.dump(results, fp)


def read_results(export_json):
    """Save results dict to a JSON-file."""
    with open(export_json, 'r') as fp:
        results = json.load(fp)
    return results


def init_svdd(hidden2, radius):
    radius_a = tf.Variable(tf.ones([1]))
    center_a = tf.Variable(tf.zeros([1, hidden2]))

    radius_s = tf.Variable(tf.ones([1])) * radius
    center_s = tf.Variable(tf.zeros([1, hidden2]))

    return radius_a, center_a, radius_s, center_s


def save_best_result(file, auc, info):
    df = pd.read_csv(file)
    old_auc = float(df["auc"][df.shape[0] - 1])
    if auc >= old_auc:
        line = pd.Series(info)
        df = df.append(line, ignore_index=True)
        print(df)
        df.to_csv(file, index=False, sep=',')

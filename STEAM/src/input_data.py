import random

import scipy.sparse as sp
import scipy.io
from scipy.io import loadmat
import inspect
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import pickle
from preprocessing import preprocess_graph, sparse_to_tuple

flags = tf.app.flags
FLAGS = flags.FLAGS


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data2(data_source):
    data = scipy.io.loadmat("../data/{}/{}.mat".format(data_source, data_source))
    labels = data["Label"]

    attr_ = data["Attributes"]
    attributes = sp.csr_matrix(attr_)
    network = sp.lil_matrix(data["Network"])

    return network, attributes, labels


def load_data3(data_source, num_true):
    content = pd.read_csv("../data/" + data_source + "/content.csv", header=None)
    feature = content.iloc[:, :].values

    struct = pd.read_csv("../data/" + data_source + "/struct.csv", header=None)
    adj = struct.as_matrix(columns=None)

    permutation = pd.read_csv("../data/" + data_source + "/permutation.csv", header=None)
    permutation = np.asarray(permutation)
    labels = []
    for i in permutation:
        label = []
        if int(i) > num_true:
            label.append(1)
        else:
            label.append(0)
        labels.append(label)

    labels = np.array(labels)
    adj = sp.lil_matrix(adj)
    feature = sp.csr_matrix(feature)

    return adj, feature, labels


def load_data1(data_source):
    content = pd.read_csv("../data/" + data_source + "/" + data_source + ".content", sep='\t', header=None)
    feature = content.iloc[:, 1:-1].as_matrix()

    contentIdx = list(content.index)
    paperId = list(content.iloc[:, 0])
    new_paperId = []
    for x in paperId:
        new_paperId.append(str(x))
    paperId = new_paperId

    map = dict(zip(paperId, contentIdx))
    cites = pd.read_csv("../data/" + data_source + "/" + data_source + ".cites", sep='\t', header=None)
    adj = np.zeros((content.shape[0], content.shape[0]))
    for i, j in zip(cites[0], cites[1]):
        x = map[i]
        y = map[str(j)]
        adj[x][y] = adj[y][x] = 1

    end_line = content.iloc[:, -1].as_matrix()
    new_labels = []
    for i in end_line:
        label = []
        if i == "error":
            label.append(1)
        else:
            label.append(0)
        new_labels.append(label)

    labels = np.array(new_labels)
    adj = sp.lil_matrix(adj)
    feature = sp.csr_matrix(feature)

    return adj, feature, labels


def load_data(data_source):
    data = scipy.io.loadmat("../data/{}.mat".format(data_source))
    labels = data["gnd"]
    labels = data["Label"]

    attributes = sp.csr_matrix(data["X"])
    network = sp.lil_matrix(data["A"])

    return network, attributes, labels


def format_data(data_source):
    if data_source == "ACM" or data_source == "BlogCatalog" or data_source == "Flickr":
        adj, features, labels = load_data2(data_source)
    elif data_source == "cora" or data_source == "pubmed":
        adj, features, labels = load_data1(data_source)
    elif data_source == "citeseer_a":
        adj, features, labels = load_data3(data_source, 3311)
    elif data_source == "WebKB_a":
        adj, features, labels = load_data3(data_source, 877)
    elif data_source == "Amazon" or data_source == "YelpChi":
        adj, features, labels = load_data4(data_source)
    elif data_source == "parliament" or data_source == "dblp" or data_source == "hvr":
        adj, features, labels = load_data6(data_source)

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])

    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    items = [adj, num_features, num_nodes, features_nonzero, adj_norm, adj_label, features, labels]
    feas = {}
    for item in items:
        item_name = retrieve_name(item)
        feas[item_name] = item

    return feas


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var and "item" not in var_name][0]


def load_data4(data_source):
    data = loadmat('../data/' + data_source + '/' + data_source + '.mat')
    adj = sp.lil_matrix(data['homo'])
    features = sp.csr_matrix(data['features'])
    labels = data['label']

    new_labels = []
    for i in labels[0]:
        xxx = []
        xxx.append(i)
        new_labels.append(xxx)
    labels = np.array(new_labels)

    return adj, features, labels


def load_data5(data_source):
    f = open('../data/' + data_source + '/node_features.pkl', "rb")
    features = sp.csr_matrix(pickle.load(f))

    f1 = open('../data/' + data_source + '/edges.pkl', "rb")
    edges = pickle.load(f1)
    adj = sp.lil_matrix(
        edges["b-s"].A + edges["s-b"].A + edges["l-b"].A + edges["b-l"].A + edges["u-b"].A + edges["b-u"].A)

    f2 = open('../data/' + data_source + '/labels.pkl', "rb")
    labels = pickle.load(f2)
    print("111")


def load_data6(data_source):
    features = pd.read_csv("../data/" + data_source + "/now/" + data_source + ".fea", sep="\t", header=None)
    features = features.iloc[:, 1:].as_matrix()

    adj = np.zeros(shape=(features.shape[0], features.shape[0]))
    adj_file = np.array(pd.read_csv("../data/" + data_source + "/now/" + data_source + ".adj", sep="\t", header=None))
    for i in range(len(adj_file)):
        line = adj_file[i]
        adj[int(line[0])][int(line[1])] = 1

    labels = np.zeros(shape=(features.shape[0], 1))
    if data_source == "parliament":
        for i in range(451, features.shape[0]):
            labels[i][0] = 1
    elif data_source == "dblp":
        for i in range(39767, features.shape[0]):
            labels[i][0] = 1

    adj = sp.lil_matrix(adj)
    features = sp.csr_matrix(features)
    return adj, features, labels

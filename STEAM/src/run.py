import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""
from pre_import import *

import tensorflow as tf
import numpy as np
from anomaly_detection import STEAMRunner
from utils import *
from tensorboardX import SummaryWriter

flags = tf.app.flags
FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

embed_dim = 128
print("### embed_dim=", embed_dim)

flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', embed_dim * 2, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', embed_dim, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 1, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 100, 'number of iterations.')
flags.DEFINE_float('alpha', 0.8, 'balance parameter')  # for attribute cost
flags.DEFINE_float('eta', 0, 'balance parameter')  # for attribute
flags.DEFINE_float('theta', 0, 'balance parameter')  # for structure
flags.DEFINE_float('beta', 0, 'balance parameter')  # for structure
flags.DEFINE_float('radius', 0, 'balance parameter')  # for structure
flags.DEFINE_float('auc', 0, 'auc')  # for structure

seeds = [1, 3, 5, 7, 9]
# seeds = [1]
aucs = []

# data_list = ['BlogCatalog', 'Flickr', 'ACM','cora','citeseer_a','pubmed','WebKB_a','Amazon','YelpChi',"parliament","dblp"，“hvr”]
# data_list = ['BlogCatalog', 'Flickr', 'ACM','cora','citeseer_a','pubmed']
data_list = ['cora']

# eta_list = np.arange(1, 10, 2).astype(np.int)
eta_list = [1]

# theta_list = np.arange(1, 101, 10).astype(np.int)
theta_list = [1]

# beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
beta_list = [0.1]

# radius_list = [0, 0.5, 1, 2, 3, 4, 5, 6, 10]
radius_list = [0]

# alpha_list = [0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
alpha_list = [0.7]

# embed_dims = [32, 64, 128, 256, 512, 1024]
# embed_dims = [128, 256, 512, 1024]
embed_dims = [128]
# embed_dims = [8,16,32,64,128,256,512,1024]

FLAGS.iterations = 50

decoder_act = [tf.nn.sigmoid, lambda x: x]

model = 'STEAM'
task = 'anomaly_detection'

for dataset_str in data_list:
    if dataset_str == 'BlogCatalog':
        eta_list = [5]
        theta_list = [40]
        decoder_act = [tf.nn.sigmoid, lambda x: x]
        beta_list = [0.01]
        alpha_list = [0.7]
        FLAGS.iterations = 80
    elif dataset_str == 'Flickr':
        eta_list = [8]
        theta_list = [90]
        decoder_act = [tf.nn.sigmoid, lambda x: x]
        beta_list = [0.00001]
        alpha_list = [0.8]
        FLAGS.iterations = 42

    elif dataset_str == 'ACM':
        eta_list = [3]
        theta_list = [10]
        decoder_act = [tf.nn.sigmoid, lambda x: x]
        beta_list = [0.09]
        alpha_list = [0.7]
        FLAGS.iterations = 67
    elif dataset_str == 'cora':
        eta_list = [1]
        theta_list = [1]
        beta_list = [0.05]
        alpha_list = [0.005]
        decoder_act = [tf.nn.sigmoid, lambda x: x]
        FLAGS.iterations = 400
    elif dataset_str == 'citeseer_a':
        eta_list = [6.5]
        theta_list = [1]
        beta_list = [0.08]
        decoder_act = [tf.nn.sigmoid, lambda x: x]
        alpha_list = [0.7]
        FLAGS.iterations = 30
    elif dataset_str == 'pubmed':
        eta_list = [1]
        theta_list = [1]
        beta_list = [0.05]
        alpha_list = [0.7]
        decoder_act = [tf.nn.sigmoid, lambda x: x]
        FLAGS.radius = 0
        FLAGS.iterations = 50
    elif dataset_str == 'WebKB_a':
        eta_list = [8]
        theta_list = [90]
        decoder_act = [tf.nn.sigmoid, lambda x: x]
        FLAGS.iterations = 50
    elif dataset_str == 'Amazon':
        eta_list = [1]
        theta_list = [1]
        alpha_list = [0.9]
        decoder_act = [tf.nn.sigmoid, lambda x: x]
        beta_list = [0.001]
        FLAGS.radius = 0
        FLAGS.learning_rate = 0.008
        FLAGS.iterations = 14
    elif dataset_str == 'YelpChi':
        eta_list = [5]
        theta_list = [20]
        alpha_list = [0.5]
        decoder_act = [tf.nn.sigmoid, lambda x: x]
        beta_list = [0.01]
        FLAGS.radius = 0
        FLAGS.learning_rate = 0.01
        FLAGS.iterations = 50
    elif dataset_str == 'parliament':
        eta_list = [1]
        theta_list = [1]
        alpha_list = [0.7]
        decoder_act = [tf.nn.sigmoid, lambda x: x]
        beta_list = [0.5]
        FLAGS.radius = 0
        FLAGS.iterations = 12
    # else:
    #     print("[ERROR] no such dataset: {}".format(dataset_str))
    #     continue

    for eta in eta_list:
        for theta in theta_list:
            for alpha in alpha_list:
                for ed in embed_dims:
                    for bt in beta_list:
                        for rrr in radius_list:
                            for seed in seeds:
                                np.random.seed(seed)
                                tf.set_random_seed(seed)
                                FLAGS.seed = seed
                                FLAGS.radius = rrr
                                FLAGS.beta = bt
                                FLAGS.hidden2 = ed
                                FLAGS.hidden1 = ed * 2
                                FLAGS.eta = eta
                                FLAGS.theta = theta
                                FLAGS.alpha = alpha

                                info = {
                                    "discriminator_out": FLAGS.discriminator_out,
                                    "discriminator_learning_rate": FLAGS.discriminator_learning_rate,
                                    "learning_rate": FLAGS.learning_rate,
                                    "hidden1": FLAGS.hidden1,
                                    "hidden2": FLAGS.hidden2,
                                    "weight_decay": FLAGS.weight_decay,
                                    "dropout": FLAGS.dropout,
                                    "features": FLAGS.features,
                                    "seed": FLAGS.seed,
                                    "iterations": FLAGS.iterations,
                                    "alpha": FLAGS.alpha,
                                    "eta": FLAGS.eta,
                                    "theta": FLAGS.theta,
                                    "beta": FLAGS.beta,
                                    "radius": FLAGS.radius
                                }
                                print(info)

                                settings = {'data_name': dataset_str,
                                            'iterations': FLAGS.iterations,
                                            'model': model,
                                            'decoder_act': decoder_act}

                                results_dir = os.path.sep.join(['results', dataset_str, task, model])
                                log_dir = os.path.sep.join(
                                    ['logs', dataset_str, task, model, '{}_{}_{}'.format(eta, theta, alpha)])

                                if not os.path.exists(results_dir):
                                    os.makedirs(results_dir)

                                if not os.path.exists(log_dir):
                                    os.makedirs(log_dir)

                                file2print = '{}/{}_{}_{}_{}_{}.json'.format(results_dir, dataset_str,
                                                                             eta, theta, alpha, embed_dim)

                                runner = None
                                if task == 'anomaly_detection':
                                    runner = STEAMRunner(settings)

                                writer = SummaryWriter(log_dir)

                                runner.erun(writer)

                                aucs.append(FLAGS.auc)

                            with open("./ex.txt", "a", encoding='utf-8') as file:
                                str1 = str(dataset_str) + ":" + str(aucs) + "," + str(np.mean(aucs)) + "," + str(
                                    np.std(aucs))
                                file.write(str1)
                                file.write("\n")
                            aucs = []

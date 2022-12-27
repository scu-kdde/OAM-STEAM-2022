from __future__ import division
from __future__ import print_function
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ""

from constructor import get_placeholder, update
from input_data import format_data
from sklearn.metrics import roc_auc_score, average_precision_score
from model import *
from optimizer import *
from utils import init_svdd, save_best_result
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


def precision_AT_K(actual, predicted, k, num_anomaly):
    act_set = np.array(actual[:k])
    pred_set = np.array(predicted[:k])
    ll = act_set & pred_set
    tt = np.where(ll == 1)[0]
    prec = len(tt) / float(k)
    rec = len(tt) / float(num_anomaly)
    return round(prec, 4), round(rec, 4)


class STEAMRunner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']
        self.decoder_act = settings['decoder_act']

    def erun(self, writer):
        model_str = self.model
        feas = format_data(self.data_name)

        print("feature number: {}".format(feas['num_features']))
        placeholders = get_placeholder()

        num_features = feas['num_features']
        features_nonzero = feas['features_nonzero']
        num_nodes = feas['num_nodes']

        radius_a, center_a, radius_s, center_s = init_svdd(FLAGS.hidden2, FLAGS.radius)

        if model_str == 'Dominant':
            model = GCNModelAE(placeholders, num_features, features_nonzero)
            opt = OptimizerAE(preds_attribute=model.attribute_reconstructions,
                              labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                              preds_structure=model.structure_reconstructions,
                              labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=FLAGS.alpha)

        elif model_str == 'STEAM':
            model = STEAM(placeholders, num_features, num_nodes, features_nonzero, self.decoder_act)
            opt = OptimizerDAE(preds_attribute=model.attribute_reconstructions,
                               labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                               preds_structure=model.structure_reconstructions,
                               labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=FLAGS.alpha,
                               eta=FLAGS.eta, theta=FLAGS.theta, beta=FLAGS.beta,
                               embeddings_a=model.embeddings_a,
                               embeddings_s=model.embeddings_s,
                               radius_a=radius_a, center_a=center_a,
                               radius_s=radius_s, center_s=center_s)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()

        ra1, ca1, rs1, cs1 = sess.run([radius_a, center_a, radius_s, center_s])
        print("radius_a", ra1)
        print("center_a", ca1)
        print("radius_s", rs1)
        print("center_s", cs1)

        for epoch in range(1, self.iteration + 1):

            train_loss, loss_struc, loss_attr, rec_error = update(model, opt, sess,
                                                                  feas['adj_norm'],
                                                                  feas['adj_label'],
                                                                  feas['features'],
                                                                  placeholders, feas['adj'])

            if epoch % 1 == 0:
                y_true = [label[0] for label in feas['labels']]
                y_true_1 = []
                for x in y_true:
                    if x == 1:
                        y_true_1.append(0)
                    else:
                        y_true_1.append(1)

                auc = 0
                ap = 0
                try:
                    scores = np.array(rec_error)
                    scores = (scores - np.min(scores)) / (
                            np.max(scores) - np.min(scores))

                    auc = roc_auc_score(y_true, scores)

                    ap = average_precision_score(y_true, scores)

                except Exception as e:
                    print("[ERROR] for auc calculation!!!")

                FLAGS.auc = auc
                print("Epoch:", '%04d' % (epoch),
                      "AUC={:.5f}".format(round(auc, 4)),
                      "AP={:.5f}".format(round(ap, 4)),
                      "train_loss={:.5f}".format(train_loss),
                      "loss_struc={:.5f}".format(loss_struc),
                      "loss_attr={:.5f}".format(loss_attr))

                writer.add_scalar('loss_total', train_loss, epoch)
                writer.add_scalar('loss_struc', loss_struc, epoch)
                writer.add_scalar('loss_attr', loss_attr, epoch)
                writer.add_scalar('auc', auc, epoch)

                info = {
                    "auc": auc,
                    "ap": ap,
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
                logfile = "./logs/" + self.data_name + ".csv"

                save_best_result(logfile, auc, info)

        ra, ca, rs, cs = sess.run([radius_a, center_a, radius_s, center_s])
        print("radius_a", ra)
        print("center_a", ca)
        print("radius_s", rs)
        print("center_s", cs)

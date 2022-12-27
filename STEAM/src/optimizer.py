import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds_attribute, labels_attribute, preds_structure, labels_structure, alpha):
        diff_attribute = tf.square(preds_attribute - labels_attribute)
        self.attribute_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_attribute, 1))
        self.attribute_cost = tf.reduce_mean(self.attribute_reconstruction_errors)
        diff_structure = tf.square(preds_structure - labels_structure)
        self.structure_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_structure, 1))
        self.structure_cost = tf.reduce_mean(self.structure_reconstruction_errors)
        self.reconstruction_errors = tf.multiply(alpha, self.attribute_reconstruction_errors) + \
                                     tf.multiply(1 - alpha, self.structure_reconstruction_errors)
        self.cost = alpha * self.attribute_cost + (1 - alpha) * self.structure_cost
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)


class OptimizerDAE(object):
    def __init__(self, preds_attribute, labels_attribute, preds_structure, labels_structure,
                 alpha, eta, theta, beta,
                 embeddings_a, embeddings_s, radius_a, center_a, radius_s, center_s):
        self.preds_attribute = preds_attribute
        self.labels_attribute = labels_attribute
        dist_a = tf.subtract(embeddings_a, center_a)
        dist_a_square = dist_a ** 2
        dist_a_norm1 = tf.norm(dist_a_square, ord=1, axis=1)
        self.svdd_a_errors = tf.abs(tf.subtract(dist_a_norm1, radius_a ** 2))
        dist_s = tf.subtract(embeddings_s, center_s)
        dist_s_square = dist_s ** 2
        dist_s_norm1 = tf.norm(dist_s_square, ord=1, axis=1)
        self.svdd_s_errors = tf.abs(tf.subtract(dist_s_norm1, radius_s ** 2))
        B_attr = labels_attribute * (eta - 1) + 1
        diff_attribute = tf.square(tf.subtract(preds_attribute, labels_attribute) * B_attr)
        self.attribute_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_attribute, 1))
        self.attribute_cost = tf.reduce_mean(self.attribute_reconstruction_errors)
        B_struc = labels_structure * (theta - 1) + 1
        diff_structure = tf.square(tf.subtract(preds_structure, labels_structure) * B_struc)
        self.structure_reconstruction_errors = (1 - beta) * tf.sqrt(
            tf.reduce_sum(diff_structure, 1)) + beta * self.svdd_s_errors
        self.structure_cost = tf.reduce_mean(self.structure_reconstruction_errors)
        self.reconstruction_errors = tf.multiply(alpha, self.attribute_reconstruction_errors) \
                                     + tf.multiply(1 - alpha, self.structure_reconstruction_errors)
        self.cost = alpha * self.attribute_cost + (1 - alpha) * self.structure_cost
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)

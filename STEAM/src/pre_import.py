import os, sys

lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

import matplotlib

matplotlib.use('Agg')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

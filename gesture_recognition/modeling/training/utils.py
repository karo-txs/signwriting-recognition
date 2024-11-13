import tensorflow as tf
import numpy as np
import random
import os


def set_seed(seed_value=42):
    # Definir seed para os diferentes módulos
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
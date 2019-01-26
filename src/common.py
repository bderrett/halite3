import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import sys
import logging

sys.modules["pandas"] = None
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

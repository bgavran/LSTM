import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

# throws segfault if numpy and matplotlib are not imported before tensorflow!
# not sure if there's a better solution than separating the imports into a separate file
# https://github.com/tensorflow/tensorflow/issues/2034

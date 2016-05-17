import time

from src.models import *
from src.utils import *

print("Started...   ")


class NNInterface:
    def __init__(self):
        self.data = DataSets(DataPath.train_path, DataPath.test_path)
        print("Data loaded.")

    def evaluate_hyperparameters(self, hyp_param):
        start_time = time.time()
        with tf.variable_scope(str(hash(frozenset(hyp_param.items())) % 10000)):
            net = RNN(hyp_param, Observer([RNNOutputConsole(5), CostConsole(2), HiddenLayerConsole()]))

            net.train(self.data, hyp_param["training_iters"], hyp_param["batch_size"],
                      display_step=hyp_param["display_step"])

            net.test(self.data, hyp_param["batch_size"])

        print("\nExecution time: {:.2f}s".format(time.time() - start_time))
        pretty_print(hyp_param)


hyperparameters_all = {"n_input": [300],
                       "n_time_steps": [7],
                       "n_hidden": [15],
                       "learning_rate": [0.001],
                       "n_classes": [2],
                       "training_iters": [50000],
                       "batch_size": [1],
                       "display_step": [1]}

nn = NNInterface()
for hyperparameters in cartesian_product(hyperparameters_all):
    print("Evaluating hyperparameters:\n", hyperparameters)
    nn.evaluate_hyperparameters(hyperparameters)

import time

from src.models import *
from src.utils import *

print("Started...   ")


class NNInterface:
    def __init__(self):
        self.data = DataSets(DataPath.train_path, DataPath.test_path)
        self.net = None
        print("Data loaded.")

    def evaluate_hyperparameters(self, hyp_param):
        start_time = time.time()
        with tf.variable_scope(str(hash(frozenset(hyp_param.items())) % 10000)):
            net = RNN(hyp_param,
                      Observer([PerformanceGraph(3), CostConsole(2), RNNOutputConsole(5)]))

            net.train(self.data, hyp_param["training_iters"], hyp_param["batch_size"],
                      display_step=hyp_param["display_step"])

            net.test(self.data, hyp_param["batch_size"])

        print("\nExecution time: {:.2f}s".format(time.time() - start_time))
        pretty_print(hyp_param)

    def visualize_neuron(self, hyp_param, model_path, batch_size):
        with tf.variable_scope("6219"):
            self.net = RNN(hyp_param)
        n_steps, n_neurons = hyp_param["n_time_steps"] - 1, hyp_param["n_hidden"]
        hx_input, hy_input = self.data.train.next_batch(batch_size, self.net.n_steps)
        hidden_feed_dict = {self.net.x: hx_input,
                            self.net.istate: np.zeros((batch_size, self.net.NN_type_koef * self.net.n_hidden))}
        hidden = self.net.run_function(model_path, self.net.hidden_l, hidden_feed_dict)
        hidden = np.array([hidden]).T
        texts = [DataFeed.w2v_to_text(hx_input[i][:-1]) for i in reversed(range(batch_size))]
        for i in range(n_neurons):
            axs = sns.heatmap(hidden[i, :, :], cbar=False, yticklabels=False, xticklabels=False, square=True)
            overlay_text(axs, texts, n_steps, batch_size)
            plt.savefig(os.path.join(DataPath.base, "visualization", "neurons" + str(batch_size), "neuron_" + str(i)),
                        dpi=600)
            print("Saved figure number " + str(i + 1) + ".")
            plt.cla()


# TODO dodaj opadajući learning rate
hyperparameters_all = {"n_input": [300],
                       "n_time_steps": [100],
                       "n_hidden": [100],
                       "learning_rate": [0.0003],
                       "n_classes": [2],
                       "training_iters": [1000000],
                       "batch_size": [1000],
                       "display_step": [10]}

nn = NNInterface()
# nn.visualize_neuron(cartesian_product(hyperparameters_all)[0], "trained_models/model.tmp", 25)

for hyperparameters in cartesian_product(hyperparameters_all):
    print("Evaluating hyperparameters:\n", hyperparameters)
    nn.evaluate_hyperparameters(hyperparameters)

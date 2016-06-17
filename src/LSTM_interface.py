import time
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm

from src.models import *
from src.utils import *

print("Started...")


class NNInterface:
    def __init__(self, hyp_param):
        self.net = RNN(hyp_param,
                       Observer([PerformanceGraph(), CostConsole(3)]))
        print("NN initialized.")

    def evaluate_hyperparameters(self, data, hyp_param):
        start_time = time.time()
        self.net.train(data, hyp_param["training_iters"], hyp_param["batch_size"],
                       display_step=hyp_param["display_step"])

        valid = self.net.validate(data, -1, self.net.model_path)

        print("\nExecution time: {:.2f}s".format(time.time() - start_time))
        pretty_print(hyp_param)
        return valid

    def visualize_neurons(self, data, hyp_param, run_path, **kwargs):
        batch_size = kwargs.get("batch_size", hyp_param["batch_size"])
        n_steps, n_neurons = hyp_param["n_time_steps"] - 1, hyp_param["n_hidden"]
        hx_input, hy_input = data.test.next_batch(batch_size, self.net.n_steps)
        hidden_feed_dict = {self.net.x: hx_input}

        hidden = self.net.run_function(run_path + ".tmp", self.net.hidden_l, hidden_feed_dict)
        hidden = np.array(hidden).T

        texts = [DataFeedw2v.w2v_to_text(hx_input[i][:-1]) for i in reversed(range(batch_size))]
        fig, ax = plt.subplots(figsize=(16, 11))
        for i in range(n_neurons):
            axs = sns.heatmap(hidden[i, :, :], cbar=False, yticklabels=False, xticklabels=False, square=True, ax=ax)
            overlay_text(axs, texts, n_steps, batch_size)

            folder_path = os.path.join(run_path, "visualization", "neurons" + str(batch_size))
            PerformanceGraph.save_image(folder_path, "neuron_" + str(i), 1000)
            plt.cla()

    def prediction_split_input(self, data, hyp_param, run_path, **kwargs):
        batch_size = kwargs.get("batch_size", hyp_param["batch_size"])
        hx_input, hy_input = data.test.next_batch(batch_size, self.net.n_steps)
        hidden_feed_dict = {self.net.x: hx_input, self.net.y: hy_input}
        y_pred = self.net.run_function(run_path + ".tmp", self.net.prediction, hidden_feed_dict)
        y_true = np.argmax(hy_input, 1)
        conf_mat = confusion_matrix(y_true, y_pred)
        correct_pred = np.equal(y_pred, y_true)
        # Tensorflow doesn't allow the "is True" boolean comparison in python
        print("Converting input to text...", end="")
        correct_input = [DataFeedw2v.w2v_to_text(hx_input[i]) for i, item in enumerate(correct_pred) if item == 1]
        incorrect_input = [DataFeedw2v.w2v_to_text(hx_input[i]) for i, item in enumerate(correct_pred) if item == 0]
        correct_text = "\n----------\n".join([" ".join(i) for i in correct_input])
        incorrect_text = "\n----------\n".join([" ".join(i) for i in incorrect_input])
        with open(os.path.join(run_path, "correctly_classified.txt"), "w+") as f:
            f.write(correct_text)
        with open(os.path.join(run_path, "incorrectly_classified.txt"), "w+") as f:
            f.write(incorrect_text)
        print("Done.")


data = DataSets(DataFeedw2v)

hyperparameters_all = {"n_input": [data.train.input_size],
                       "n_time_steps": [50, 150],
                       "n_layers": [2, 3],
                       "n_hidden": [100, 500],
                       "starting_learning_rate": [0.001, 0.0001],
                       "decay_rate": [0.6321],
                       "decay_steps_div": [10],
                       "n_classes": [2],
                       "training_iters": [500000],
                       "batch_size": [1000],
                       "display_step": [10],
                       "n_tries": [2]}

# model_path = "tf_logs/2016_May_22__14:18"
# path = os.path.join(DataPath.base, model_path)
maxx = []
for i, hyperparameters in enumerate(cartesian_product(hyperparameters_all)):
    maxx.append(0)
    for _ in range(hyperparameters["n_tries"]):
        print("Evaluating hyperparameters:\n", hyperparameters)
        nn = NNInterface(hyperparameters)
        result = nn.evaluate_hyperparameters(data, hyperparameters)
        if result > maxx[-1]:
            maxx[-1] = result
        # path + ".tmp
        # nn.prediction_split_input(data, hyperparameters, nn.net.tb_path, batch_size=100)
        # nn.visualize_neurons(data, hyperparameters, path, batch_size=40)
        tf.reset_default_graph()
        print("Current state:")
        print(maxx)
best_hyperparameters = cartesian_product(hyperparameters_all)[np.argmax(maxx)]
print("Best hyperparameters:")
print(best_hyperparameters)
nn = NNInterface(best_hyperparameters)
nn.evaluate_hyperparameters(data, best_hyperparameters)
nn.net.test(data, -1, nn.net.model_path)

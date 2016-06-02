import time
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import svm

from src.models import *
from src.utils import *

print("Started...")


class NNInterface:
    def __init__(self, hyp_param):
        self.net = RNN(hyp_param,
                       Observer([PerformanceGraph(), CostConsole(3), RNNOutputConsole(5)]))
        print("NN initialized.")

    def evaluate_hyperparameters(self, data, hyp_param):
        start_time = time.time()
        self.net.train(data, hyp_param["training_iters"], hyp_param["batch_size"],
                       display_step=hyp_param["display_step"])

        self.net.test(data, 25000, self.net.model_path)

        print("\nExecution time: {:.2f}s".format(time.time() - start_time))
        pretty_print(hyp_param)

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
        print("Converting input to text...")
        correct_input = [DataFeedw2v.w2v_to_text(hx_input[i]) for i, item in enumerate(correct_pred) if item == 1]
        incorrect_input = [DataFeedw2v.w2v_to_text(hx_input[i]) for i, item in enumerate(correct_pred) if item == 0]
        correct_text = "\n----------\n".join([" ".join(i) for i in correct_input])
        incorrect_text = "\n----------\n".join([" ".join(i) for i in incorrect_input])
        with open(os.path.join(run_path, "correctly_classified.txt"), "w+") as f:
            f.write(correct_text)
        with open(os.path.join(run_path, "incorrectly_classified.txt"), "w+") as f:
            f.write(incorrect_text)


s = len(DataFeedOneHot.vec.vocabulary_)

hyperparameters_all = {"n_input": [300],
                       "n_time_steps": [100],
                       "n_layers": [1],
                       "n_hidden": [100],
                       "starting_learning_rate": [0.001],
                       "decay_rate": [0.6321],
                       "decay_steps_div": [10],
                       "n_classes": [2],
                       "training_iters": [300000],
                       "batch_size": [1000],
                       "display_step": [10]}

data = DataSets(DataFeedw2v)
############################################################
# batch_size = 500
# start_time = time.time()
# clf = svm.SVC()
# train_x, train_y = data.train.next_batch(batch_size, 100, sparse_vector=True)
# test_x, test_y = data.test.next_batch(batch_size, 100, sparse_vector=True)
#
# print("Data loaded.")
#
# train_x = np.array(train_x).reshape(batch_size, -1)
# test_x = np.array(test_x).reshape(batch_size, -1)
#
# print("Data transformed.")
#
# # train_x = sparse.csr_matrix(train_x)
# # test_x = sparse.csr_matrix(test_x)
#
# print("Sparse vectors created.")
#
# train_y = [1 if i == [1, 0] else 0 for i in train_y]
# test_y = [1 if i == [1, 0] else 0 for i in test_y]
# print("Training...")
# clf.fit(train_x, train_y)
# print("Accuracy score:", end="\n")
# acc = accuracy_score(test_y, clf.predict(test_x))
# print(acc)
# print("\nExecution time: {:.2f}s".format(time.time() - start_time))
# input()

############################################################

# model_path = "tf_logs/2016_May_22__14:18"
# path = os.path.join(DataPath.base, model_path)
for hyperparameters in cartesian_product(hyperparameters_all):
    print("Evaluating hyperparameters:\n", hyperparameters)
    nn = NNInterface(hyperparameters)
    nn.evaluate_hyperparameters(data, hyperparameters)
    # path + ".tmp
    # nn.net.test(data, 25000, nn.net.model_path)
    nn.prediction_split_input(data, hyperparameters, nn.net.tb_path, batch_size=100)
    # nn.visualize_neurons(data, hyperparameters, path, batch_size=40)
    tf.reset_default_graph()

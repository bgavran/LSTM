from time import gmtime, strftime

from src.tf_imports import *
from src.utils import *


class Observer:
    def __init__(self, observers):
        self.observers = [i for i in observers]

    def add_observer(self, observer):
        self.observers.append(observer)

    def remove_observer(self, observer):
        self.observers.remove(observer)

    def notify(self, nn, *args, **kwargs):
        for observer in self.observers:
            observer.update(nn, *args, **kwargs)


class HiddenLayerGraph:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.fig.show()

    def update(self, nn, *args, **kwargs):
        self.fig.canvas.draw_idle()
        plt.pause(0.0001)
        hidden_layer = np.array([i[0] for i in kwargs["hidden_output"]]).T
        sns.heatmap(hidden_layer, cbar=False, square=True, annot=False, ax=self.ax)
        self.ax.set_xlabel("Time steps")
        self.ax.set_ylabel("Neuron")
        print("lol")


class PerformanceGraph:
    def __init__(self, n_metrics):
        self.perf_data = [[] for _ in range(n_metrics)]
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()

        sns.set_style("darkgrid")
        self.pallete = sns.color_palette("bright")
        self.fig.show()

    def update(self, nn, *args, **kwargs):
        self.fig.canvas.draw_idle()
        plt.pause(0.0001)
        batch_size = kwargs["batch_size"]
        display_step = kwargs["display_step"]
        for i, data in enumerate(self.perf_data):
            data.append(kwargs["plot_data"][i])
        self.ax1.set_xlabel("Training iterations (x " + str(batch_size * display_step) + ")" +
                            "\nLearning rate " + str(nn.learning_rate) +
                            "\nHidden layer size " + str(nn.n_hidden) +
                            "\nBatch size: " + str(batch_size) +
                            "\nTime steps: " + str(nn.n_steps))
        self.plot_data()
        if kwargs.get("save_fig", 0):
            self.fig.savefig(os.path.join(DataPath.base, "logs" + strftime("%Y_%B_%d__%H:%M", gmtime())), dpi=300)

    def plot_data(self):
        col = self.pallete[2]

        x_range = range(len(self.perf_data[0]))
        line1 = self.ax1.semilogy(x_range, self.perf_data[0], color=col, label="Train loss")
        self.ax1.set_ylabel('Train loss', color=col)
        for tl in self.ax1.get_yticklabels():
            tl.set_color(col)

        col = self.pallete[0]
        line2 = self.ax2.plot(x_range, self.perf_data[1], color=col, label="Train accuracy")
        self.ax2.set_ylabel("Train/test accuracy", color=col)
        for tl in self.ax2.get_yticklabels():
            tl.set_color(col)

        line3 = self.ax2.plot(x_range, self.perf_data[2], color=self.pallete[3], label="Test accuracy")

        lns = line1 + line2 + line3
        labs = [l.get_label() for l in lns]
        self.ax1.legend(lns, labs, loc="best")


class CostConsole:
    def __init__(self, n_metrics):
        self.perf_data = [[] for _ in range(n_metrics)]

    def update(self, nn, *args, **kwargs):
        step = kwargs["step"]
        batch_size = kwargs["batch_size"]
        for i, data in enumerate(self.perf_data):
            data.append(kwargs["plot_data"][i])
        print("Step " + str(step) + " Iter " + str(step * batch_size) +
              ", Train Loss= " + "{:.6f}".format(self.perf_data[0][-1]) +
              ", Train Accuracy= " + "{:.5f}".format(self.perf_data[1][-1]) +
              ", Test Accuracy= " + "{:.5f}".format(self.perf_data[1][-1]))


class RNNOutputConsole:
    def __init__(self, koliko):
        self.rez_list = []
        self.koliko = koliko

    def update(self, nn, *args, **kwargs):
        self.rez_list.append(kwargs["rez"])
        print("Network output:\n", self.rez_list[-1][:self.koliko])
        print("---------\n")

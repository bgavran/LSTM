from src.observer import *
from tensorflow.models.rnn import *


class RNN:
    def __init__(self, hyp_param, observers=None):
        self.global_step = tf.Variable(0, trainable=False)
        self.decay_rate, self.decay_steps_div = hyp_param["decay_rate"], hyp_param["decay_steps_div"]
        self.starting_learning_rate = hyp_param["starting_learning_rate"]
        self.learning_rate = tf.train.exponential_decay(self.starting_learning_rate,
                                                        global_step=self.global_step,
                                                        decay_steps=hyp_param["training_iters"] /
                                                                    hyp_param["batch_size"] /
                                                                    self.decay_steps_div,
                                                        decay_rate=self.decay_rate,
                                                        name="Decaying_learning_rate")
        tf.scalar_summary("learning_rate", self.learning_rate)
        self.n_input, self.n_steps, = hyp_param["n_input"], hyp_param["n_time_steps"]
        self.n_hidden, self.n_classes = hyp_param["n_hidden"], hyp_param["n_classes"]
        self.n_layers = hyp_param["n_layers"]
        self.observers = observers

        with tf.name_scope("X"):
            self.x = tf.placeholder("float", [None, self.n_steps, self.n_input], name="X")
        with tf.name_scope("Y"):
            self.y = tf.placeholder("float", [None, self.n_classes], name="Y")
        self.feed_dict = dict()

        print("Initializing network...")
        with tf.name_scope("RNN"):
            with tf.name_scope("weights"):
                self.weights = {
                    'hidden': tf.Variable(tf.random_normal([self.n_input, self.n_hidden]), name="hidden"),
                    'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]), name="out")
                }
            with tf.name_scope("biases"):
                self.biases = {'hidden': tf.Variable(tf.random_normal([self.n_hidden]), name="hidden"),
                               'out': tf.Variable(tf.random_normal([self.n_classes]), name="out")}
            with tf.name_scope("Hidden_layer"):
                self.hidden_l = self.compute_hidden(self.x, self.weights, self.biases)
            with tf.name_scope("Output"):
                self.output = tf.matmul(self.hidden_l[-1], self.weights["out"]) + self.biases['out']

        print("Initializing loss and optimizer...")
        with tf.name_scope("Cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, self.y), name="cost")
            tf.scalar_summary('cost', self.cost)
        with tf.name_scope("Optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
                .minimize(self.cost, global_step=self.global_step)

        print("Initializing evaluation...")
        with tf.name_scope("Evaluation"):
            with tf.name_scope("Correct_prediction"):
                self.output_softmax = tf.nn.softmax(self.output, name="softmax")
                self.prediction = tf.argmax(self.output_softmax, 1, name="max_value")
                self.correct_pred = tf.equal(self.prediction, tf.argmax(self.y, 1),
                                             name="correct_prediction")
            with tf.name_scope("Accuracy"):
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name="accuracy")
                tf.scalar_summary("accuracy", self.accuracy)

        self.train_writer = None
        self.merged = None

        self.timestamp = strftime("%Y_%B_%d__%H:%M", gmtime())
        self.tb_path = os.path.join(DataPath.base, "tf_logs", self.timestamp)

        self.saver = tf.train.Saver()
        self.model_path = self.tb_path + ".tmp"

    def compute_hidden(self, _x, _weights, _biases):
        with tf.name_scope("Reshaping_and_transposing"):
            batch_size = tf.shape(_x)[0]
            # input shape: (batch_size, n_steps, n_input)
            _x = tf.transpose(_x, [1, 0, 2])  # permute n_steps and batch_size
            # Reshape to prepare input to hidden activation
            _x = tf.reshape(_x, [-1, self.n_input])  # (n_steps*batch_size, n_input)
            # Linear activation
            _x = tf.matmul(_x, _weights['hidden']) + _biases['hidden']

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            _x = tf.split(0, self.n_steps, _x)  # n_steps * (batch_size, n_hidden)

        # Define a rnn cell with tensorflow
        basic_rnn_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1)
        stacked_rnn = rnn_cell.MultiRNNCell([basic_rnn_cell] * self.n_layers)

        with tf.name_scope("RNN_initial_state"):
            _istate = stacked_rnn.zero_state(batch_size, tf.float32)

        # Get rnn cell output
        outputs, states = rnn.rnn(stacked_rnn, _x, initial_state=_istate)
        return outputs

    def train(self, data, training_iters, batch_size, display_step=1):
        assert training_iters % batch_size == 0

        print("Starting computation...")
        with tf.Session() as sess:
            self.merged = tf.merge_all_summaries()
            self.train_writer = tf.train.SummaryWriter(os.path.join(self.tb_path, "train"), sess.graph)
            sess.run(tf.initialize_all_variables())
            for step in range(int(training_iters / batch_size)):
                batch_xs, batch_ys = data.train.next_batch(batch_size, self.n_steps)
                self.feed_dict = {self.x: batch_xs, self.y: batch_ys}
                sess.run(self.optimizer, feed_dict=self.feed_dict)
                if step % display_step == 0:
                    self.log(sess, data, train_data=[batch_size, step, display_step])

            print("\nOptimization Finished!")
            self.log(sess, data, end=True, train_data=[batch_size, -1, display_step])

            self.saver.save(sess, self.model_path)

    def test(self, data, batch_size, model_path):
        test_data, test_label = data.test.next_batch(batch_size, self.n_steps)
        feed_dict = {self.x: test_data, self.y: test_label}
        test_acc = self.run_function(model_path, self.accuracy, feed_dict)
        print("--------------------Test accuracy--------------------\n"
              "                       ", test_acc)

    def run_function(self, model_path, fn, feed_dict):
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            return sess.run(fn, feed_dict=feed_dict)

    def log(self, sess, data, end=False, **kwargs):
        rez = sess.run(self.output_softmax, feed_dict=self.feed_dict)
        summary, train_acc, train_loss, _ = sess.run([self.merged, self.accuracy, self.cost, self.learning_rate],
                                                     feed_dict=self.feed_dict)
        hidden_output = sess.run(self.hidden_l, feed_dict=self.feed_dict)

        self.train_writer.add_summary(summary, kwargs["train_data"][1])

        tbatch_size = kwargs["train_data"][0]
        tbatch_xs, tbatch_ys = data.test.next_batch(tbatch_size, self.n_steps)
        feed_dict = {self.x: tbatch_xs, self.y: tbatch_ys}
        test_accuracy = sess.run(self.accuracy, feed_dict=feed_dict)

        self.observers.notify(self, batch_size=kwargs["train_data"][0], step=kwargs["train_data"][1],
                              display_step=kwargs["train_data"][2], rez=rez,
                              plot_data=[train_loss, train_acc, test_accuracy], save_fig=end,
                              hidden_output=hidden_output)

from src.observer import *
from tensorflow.models.rnn import *


class RNN:
    def __init__(self, hyp_param, observers=None, lstm_flag=True):
        self.learning_rate = hyp_param["learning_rate"]
        self.n_input, self.n_steps, = hyp_param["n_input"], hyp_param["n_time_steps"]
        self.n_hidden, self.n_classes = hyp_param["n_hidden"], hyp_param["n_classes"]
        self.NN_type_koef = lstm_flag + 1  # mapping iz [0, 1] u [1, 2]. LSTM zahtijeva 2x više veza

        self.x = tf.placeholder("float", [None, self.n_steps, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])
        self.istate = tf.placeholder("float", [None, self.NN_type_koef * self.n_hidden])
        self.feed_dict = dict()

        self.weights = {
            'hidden': tf.Variable(tf.random_normal([self.n_input, self.n_hidden])),
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {'hidden': tf.Variable(tf.random_normal([self.n_hidden])),
                       'out': tf.Variable(tf.random_normal([self.n_classes]))}

        print("Initializing variables...")
        self.output = self.compute_output(self.x, self.istate, self.weights, self.biases)
        tf.get_variable_scope().reuse_variables()
        self.hidden_l = self.compute_hidden(self.x, self.istate, self.weights, self.biases)

        print("Initializing loss and optimizer...")
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        print("Initializing evaluation...")
        self.correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.saver = tf.train.Saver()
        self.save_path = os.path.join(DataPath.base, "tf_logs", dictionary_to_file_path(hyp_param),
                                      strftime("%Y_%B_%d__%H:%M", gmtime()))

        self.observers = observers

    def compute_hidden(self, _x, _istate, _weights, _biases):
        # input shape: (batch_size, n_steps, n_input)
        _x = tf.transpose(_x, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        _x = tf.reshape(_x, [-1, self.n_input])  # (n_steps*batch_size, n_input)
        # Linear activation
        _x = tf.matmul(_x, _weights['hidden']) + _biases['hidden']

        # Define a rnn cell with tensorflow
        basic_rnn_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1)

        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _x = tf.split(0, self.n_steps, _x)  # n_steps * (batch_size, n_hidden)

        # Get rnn cell output
        outputs, states = rnn.rnn(basic_rnn_cell, _x, initial_state=_istate)
        return outputs

    def compute_output(self, _x, _istate, _weights, _biases):
        outputs = self.compute_hidden(_x, _istate, _weights, _biases)

        # Linear activation
        # Get inner loop last output
        return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

    def train(self, data, training_iters, batch_size, display_step=1):
        assert training_iters % batch_size == 0

        print("Starting computation...")
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for step in range(int(training_iters / batch_size)):
                batch_xs, batch_ys = data.train.next_batch(batch_size, self.n_steps)
                self.feed_dict = {self.x: batch_xs, self.y: batch_ys,
                                  self.istate: np.zeros((batch_size, self.NN_type_koef * self.n_hidden))}
                sess.run(self.optimizer, feed_dict=self.feed_dict)
                if step % display_step == 0:
                    self.log(sess, data, train_data=[batch_size, step, display_step])

            print("\nOptimization Finished!")
            self.log(sess, data, end=True, train_data=[batch_size, -1, display_step])
            self.saver.save(sess, self.save_path)

    def test(self, data, batch_size):
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)
            test_data, test_label = data.test.next_batch(batch_size, self.n_steps)
            feed_dict = {self.x: test_data, self.y: test_label,
                         self.istate: np.zeros((batch_size, self.NN_type_koef * self.n_hidden))}
            test_acc = sess.run(self.cost, feed_dict=feed_dict)
            print("Testing Accuracy:", test_acc)

    def run_function(self, model_path, fn, feed_dict):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.saver.restore(sess, model_path)
            return sess.run(fn, feed_dict=feed_dict)

    def log(self, sess, data, end=False, **kwargs):
        rez = sess.run(tf.nn.softmax_cross_entropy_with_logits(self.output), feed_dict=self.feed_dict)
        train_acc = sess.run(self.accuracy, feed_dict=self.feed_dict)
        train_loss = sess.run(self.cost, feed_dict=self.feed_dict)
        hidden_output = sess.run(self.hidden_l, feed_dict=self.feed_dict)

        tbatch_size = kwargs["train_data"][0]
        tbatch_xs, tbatch_ys = data.test.next_batch(tbatch_size, self.n_steps)
        feed_dict = {self.x: tbatch_xs, self.y: tbatch_ys,
                     self.istate: np.zeros((tbatch_size, self.NN_type_koef * self.n_hidden))}
        test_accuracy = sess.run(self.accuracy, feed_dict=feed_dict)

        self.observers.notify(self, batch_size=kwargs["train_data"][0], step=kwargs["train_data"][1],
                              display_step=kwargs["train_data"][2], rez=rez,
                              plot_data=[train_loss, train_acc, test_accuracy], save_fig=end,
                              hidden_output=hidden_output)

    def interactive(self):
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)
            while True:
                inp = input("Enter input to be analyzed:")
                # # inp_arr = ReadFiles.to_constant_dense(ReadFiles.to_w2v(inp), self.n_steps)
                # # prediction = sess.run(tf.nn.softmax(self.pred), feed_dict={self.x: [inp_arr],
                #                                                            self.istate: np.zeros(
                #                                                                (1,
                #                                                                 self.NN_type_koef * self.n_hidden))})
                # print("Prediction: ", prediction)

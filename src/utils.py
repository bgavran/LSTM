import os
import json
import pickle
import re
import gensim
import numpy as np

from itertools import product
from sklearn.utils import shuffle


class DataPath:
    base = json.loads(open("config.json").read()).get("path", "")
    data = "data"
    imdb = "aclImdb"
    w2v_name = "myw2v.txt"
    train_p_name = "train.p"
    test_p_name = "test.p"

    base_imdb = os.path.join(base, data, imdb)
    train_path = os.path.join(base_imdb, train_p_name)
    test_path = os.path.join(base_imdb, test_p_name)
    w2v_path = os.path.join(base_imdb, w2v_name)


class DataFeed:
    word2vec_dict = gensim.models.Word2Vec.load_word2vec_format(DataPath.w2v_path, binary=False)

    def __init__(self, pickle_file_path):
        self.curr_batch_index = 0
        data = pickle.load(open(pickle_file_path, "rb"))
        self.x = np.array(data[0])  # reviews
        self.y = np.array(data[1])  # ratings

        # losing info on the actual file names (because the ordering is wrong)
        self.x, self.y = shuffle(self.x, self.y, random_state=0)
        self.batch_order = np.array([])

    @staticmethod
    def to_constant_dense(arr_input, review_size):
        delim = np.full((1, arr_input.shape[1]), 42, np.float64)
        if arr_input.shape[0] < review_size:
            zeroes = np.zeros((review_size - arr_input.shape[0] - 1, arr_input.shape[1]))
            ret = np.vstack((arr_input, delim, zeroes))
        else:
            ret = np.vstack((arr_input[:review_size - 1], delim))
        return ret

    def next_batch(self, batch_size, review_size):
        batch_count = len(self.x) / batch_size
        assert batch_count == int(batch_count)  # number of batches must be an integer
        batch_count = int(batch_count)
        if self.curr_batch_index >= batch_count or not self.batch_order.size:
            self.curr_batch_index = 0
            self.batch_order = np.random.permutation(batch_count)

        batch_x = np.split(self.x, batch_count)[self.batch_order[self.curr_batch_index]]
        batch_y = np.split(self.y, batch_count)[self.batch_order[self.curr_batch_index]]

        batch_x = [DataFeed.to_constant_dense(self.text_to_w2v(i), review_size) for i in batch_x]
        batch_y = [[1, 0] if int(i) > 5 else [0, 1] for i in batch_y]

        self.curr_batch_index += 1
        return batch_x, batch_y

    @staticmethod
    def text_to_w2v(review_text):
        insert_space = r"([\w])([\.,])", r"\1 \2"
        review_text = re.sub(insert_space[0], insert_space[1], review_text)
        word_embedding = []
        count = 0
        for i in review_text.split(" "):
            try:
                word_embedding.append(DataFeed.word2vec_dict[i.lower()])
            except KeyError:
                count += 1
        return np.array(word_embedding)
        # return np.vstack((word_embedding, [[0] * len(word_embedding[0])] * count))

    @staticmethod
    def w2v_to_text(time_step_vectors):
        text = []
        for i, vector in enumerate(time_step_vectors):
            if vector[0] == 42:
                text.extend(["0"] * (len(vector) - i - 1))
                break
            ind = np.where(np.all(vector == DataFeed.word2vec_dict.syn0, axis=1))[0][0]
            text.append(DataFeed.word2vec_dict.index2word[ind])
        return text


class DataSets:
    def __init__(self, train_path, test_path):
        self.train = DataFeed(train_path)
        self.test = DataFeed(test_path)


def dictionary_to_file_path(hyp):
    replacement = {"{": "", "}": "", "[": "", "]": "", "'": "", " ": "",
                   ",": "_"}
    return "".join([replacement.get(char, char) for char in str(hyp)])


def pretty_print(hyp):
    print("----------Network info----------")
    print("Input layer size: " + str(hyp["n_input"]))
    print("Hidden layer size: " + str(hyp["n_hidden"]))
    print("Output layer size: " + str(hyp["n_classes"]))
    print("")
    print("Recurrent time steps: " + str(hyp["n_time_steps"]))

    print("----------Training hyperparameters----------")
    print("Learning rate: " + str(hyp["learning_rate"]))
    print("Batch size: " + str(hyp["batch_size"]))
    print("Training iterations: " + str(hyp["training_iters"]))


def cartesian_product(dicts):
    return [dict(zip(dicts, x)) for x in product(*dicts.values())]


def overlay_text(axs, texts, n_time_steps, batch_size):
    x = np.arange(0, n_time_steps)
    y = np.arange(0, batch_size)
    grid = np.dstack(np.meshgrid(x, y))
    x_offset = 0.1
    y_offset = 0.4
    grid = [[[elem[0] + x_offset, elem[1] + y_offset] for elem in row] for row in grid]
    for i, row in enumerate(grid):
        for j, elem in enumerate(row):
            axs.text(elem[0], elem[1], texts[i][j], fontsize=(4 - len(texts[i][j])))
    return axs

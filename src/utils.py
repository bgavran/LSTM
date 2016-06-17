import os
import json
import pickle
import re
import gensim
import numpy as np

from itertools import product
from sklearn.utils import shuffle
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from natsort import natsorted


class DataPath:
    base = json.loads(open("config.json").read()).get("path", "")
    data = "data"
    imdb = "aclImdb"
    w2v_name = "myw2v.txt"
    train_p_name = "train.p"
    test_p_name = "test.p"
    vocab_name = "imdb.vocab"

    base_imdb = os.path.join(base, data, imdb)
    vocab_path = os.path.join(base_imdb, vocab_name)
    train_path = os.path.join(base_imdb, "train")
    test_path = os.path.join(base_imdb, "test")
    train_p_path = os.path.join(base_imdb, train_p_name)
    test_p_path = os.path.join(base_imdb, test_p_name)
    w2v_path = os.path.join(base_imdb, w2v_name)


class Data:
    def next_batch(self, batch_size, review_size):
        raise NotImplementedError()

    @staticmethod
    def to_constant_dense(arr_input, review_size):
        delim = np.full((1, arr_input.shape[1]), 42, np.float64)
        if arr_input.shape[0] < review_size:
            zeroes = np.zeros((review_size - arr_input.shape[0] - 1, arr_input.shape[1]))
            # app = np.vstack((delim, zeroes))
            # ret = vstack((arr_input, app))
            ret = np.vstack((arr_input, delim, zeroes))
        else:
            ret = np.vstack((arr_input[:review_size - 1], delim))
        return ret


class DataFeedw2v(Data):
    input_size = 300
    word2vec_dict = gensim.models.Word2Vec.load_word2vec_format(DataPath.w2v_path, binary=False)

    def __init__(self, data):
        self.curr_batch_index = 0
        self.last_batch_size = None
        self.x = np.array(data[0])  # reviews
        self.y = np.array(data[1])  # ratings

        # losing info on the actual file names (because the ordering is wrong)
        self.x, self.y = shuffle(self.x, self.y, random_state=0)
        self.batch_order = np.array([])

    def next_batch(self, batch_size, review_size, sparse=False):
        if batch_size == -1:
            batch_size = len(self.x)
        batch_count_d = len(self.x) / batch_size
        batch_count = int(batch_count_d)
        assert batch_count_d == batch_count  # number of batches must be an integer
        if self.curr_batch_index >= batch_count or not self.batch_order.size or (
                        self.last_batch_size is not None and self.last_batch_size != batch_size):
            self.curr_batch_index = 0
            self.batch_order = np.random.permutation(batch_count)
            self.last_batch_size = batch_size

        batch_x = np.split(self.x, batch_count)[self.batch_order[self.curr_batch_index]]
        batch_y = np.split(self.y, batch_count)[self.batch_order[self.curr_batch_index]]

        batch_x = [DataFeedw2v.to_constant_dense(self.text_to_w2v(i), review_size) for i in batch_x]
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
                word_embedding.append(DataFeedw2v.word2vec_dict[i.lower()])
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
            ind = np.where(np.all(vector == DataFeedw2v.word2vec_dict.syn0, axis=1))[0][0]
            text.append(DataFeedw2v.word2vec_dict.index2word[ind])
        return text

    @staticmethod
    def read_files_pos_neg(file_path):
        review_text = []
        review_rating = []
        c = 0
        for i, file in enumerate(natsorted(os.listdir(file_path))):
            if i % 1000 == 0:
                print(c, "    ", file)
            c += 1
            # tuple = (file_text, rating)
            rating = file[-6:-4]
            if rating[0] == "_":
                rating = rating[1]
            review_text.append(open(file_path + file).read())
            review_rating.append(rating)
        return review_text, review_rating


class DataFeedOneHot(Data):
    input_size = 1000
    word_list = [i for i in open(DataPath.vocab_path).read().split()]
    vec = CountVectorizer(stop_words="english", max_features=input_size).fit(word_list)

    def __init__(self, base_path):
        self.curr_batch_index = 0
        self.last_batch_size = None
        self.pos_review, self.pos_rating = self.load_files(os.path.join(base_path, "pos"))
        self.neg_review, self.neg_rating = self.load_files(os.path.join(base_path, "neg"))

        self.pos_bow = [DataFeedOneHot.vec.transform(review.split()) for review in self.pos_review]
        self.neg_bow = [DataFeedOneHot.vec.transform(review.split()) for review in self.neg_review]

        self.reviews = np.array(self.pos_bow + self.neg_bow)
        self.ratings = np.array(self.pos_rating + self.neg_rating)

        self.x, self.y = shuffle(self.reviews, self.ratings, random_state=0)
        self.batch_order = np.array([])

    def next_batch(self, batch_size, review_size, sparse_vector=False):
        batch_count_d = len(self.x) / batch_size
        batch_count = int(batch_count_d)
        assert batch_count_d == batch_count  # number of batches must be an integer
        if self.curr_batch_index >= batch_count or not self.batch_order.size or (
                        self.last_batch_size is not None and self.last_batch_size != batch_size):
            self.curr_batch_index = 0
            self.batch_order = np.random.permutation(batch_count)
            self.last_batch_size = batch_size

        batch_x = np.split(self.x, batch_count)[self.batch_order[self.curr_batch_index]]
        batch_y = np.split(self.y, batch_count)[self.batch_order[self.curr_batch_index]]

        if sparse_vector:
            batch_x = sparse.csr_matrix([(Data.to_constant_dense(review.toarray(), review_size)) for review in batch_x])
        else:
            batch_x = [DataFeedOneHot.to_constant_dense(i.toarray(), review_size) for i in batch_x]
        batch_y = [[1, 0] if int(i) > 5 else [0, 1] for i in batch_y]

        self.curr_batch_index += 1
        return batch_x, batch_y

    @staticmethod
    def load_files(file_path):
        review_text = []
        review_rating = []
        c = 0

        for file in natsorted(os.listdir(file_path)):
            print(c, "    ", file)
            c += 1
            # tuple = (file_text, rating)
            rating = file[-6:-4]

            if rating[0] == "_":
                rating = rating[1]
            review_text.append(open(os.path.join(file_path, file)).read())
            review_rating.append(rating)

        return review_text, review_rating


class DataSets:
    def __init__(self, data_feed):
        if data_feed is DataFeedw2v:
            self.train_path = DataPath.train_p_path
            self.test_path = DataPath.test_p_path
        elif data_feed is DataFeedOneHot:
            self.train_path = DataPath.train_path
            self.test_path = DataPath.test_path
        else:
            raise Exception("Invalid data_feed argument!")

        data_train = np.array(pickle.load(open(self.train_path, "rb")))
        data_validation_test = np.array(pickle.load(open(self.test_path, "rb")))
        delim = 10000
        data_validation = data_validation_test[:, :delim]
        data_test = data_validation_test[:, delim:]
        self.train = data_feed(data_train)
        self.validation = data_feed(data_validation)
        self.test = data_feed(data_test)


def dictionary_to_file_path(hyp):
    replacement = {"{": "", "}": "", "[": "", "]": "", "'": "", " ": "", ":": "",
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
    print("Learning rate: " + str(hyp["starting_learning_rate"]))
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

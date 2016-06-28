# LSTM

Playing around with various LSTM topologies and trying to figure out TensorFlow.
Using [aclIMDb data](http://ai.stanford.edu/~amaas/data/sentiment/ "aclIMDb") to do that.

Achieved ~85% accuracy. It yields different types of activations on different neurons:

Not a very interesting neuron, it models some uninterpretable features.
![alt text](https://github.com/bgavran3/LSTM/blob/master/img/not_interesting.png "")

Neuron that fires positively when detects words such as *worst*, *bad*, *awful* and positively when it detects words such as *classic*, *good*, *finest* etc.
It also puts them on a linear scale where some words are more positive/negative than others.
![alt text](https://github.com/bgavran3/LSTM/blob/master/img/sentiment_detector.png "")

Neuron that learned to count zeroes:
![alt text](https://github.com/bgavran3/LSTM/blob/master/img/zero_counter.png "")

Inspired by Karpathy's [amazing article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/ "").
